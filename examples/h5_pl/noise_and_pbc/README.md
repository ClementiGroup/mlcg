# Training with periodic boundary conditions, noise augmentation and an on-the-fly prior

This example trains a SchNet model on a coarse-grained POPC lipid bilayer (1800 beads per frame, 175 frames). It differs from the `single_molecule` and `multiple_molecules` examples in three ways:

1. **The periodic boundary conditions (PBC) of each frame are loaded from the H5 file.**
2. **The training data is augmented with Gaussian noise.**
3. **The prior is evaluated on the fly**, while the batches are built, instead of being subtracted beforehand and stored as delta forces in the H5 file.

> **Warning: this is a demo of the code, not a production setup.** To keep the example small and practical, the prior contains only bonded terms (`bonds` and `angles`), with no non-bonded or repulsion term, and the bonded pairs are not excluded from the interactions of the network (the dataset has no `excluded_pairs`). The dataset is also tiny. A model trained with these files is not expected to be stable or accurate in simulation.

This example uses the `mlcg-train_h5_ng` script and its data module (`mlcg.pl.h5_data_ng.H5DataModule`). The `mlcg-train_h5` script used in the other examples also supports PBC, but it does not have the noise augmentation or the on-the-fly prior options.

## Files

| File | Content |
| --- | --- |
| `training_config.yaml` | Pytorch Lightning configuration: dataset, model, optimizer and trainer |
| `bilayer_with_pbc.h5` | Dataset. Metaset `POPC`, molecule `bilayer`, with `cg_coords`, `cg_forces`, `cell`, `pbc` and the attribute `cg_embeds` |
| `bilayer_partition.yaml` | Train/validation split and batch sizes |
| `bilayer_prior.pt` | Prior model: a `SumOut` with `bonds` and `angles` terms |
| `bilayer_nls.pt` | Neighbor lists of the prior terms (`bonds`, `angles`) and the force map `fmap` |

## Running the training

From this folder:

```bash
mlcg-train_h5_ng fit --config ./training_config.yaml
```

The checkpoints are written to `./ckpt` and the tensorboard logs to `./tensorboard`. The configuration sets `ckpt_path: last`, so running the command again resumes from `./ckpt/last.ckpt` if it exists.

The prior is evaluated on the GPU inside the data loader, so a CUDA device is required.

## 1. Loading the PBC options

The cell and the periodicity flags are stored per frame in the H5 file: `cell` has shape `(n_frames, 3, 3)` and `pbc` has shape `(n_frames, 3)`. Two settings in the `data` section are needed to bring them into the training batches:

```yaml
data:
  with_pbc: true
  loading_options:
    hdf_key_mapping:
      coords: cg_coords
      forces: cg_forces
      embeds: attrs:cg_embeds
      pbc: pbc
      cell: cell
```

- The `pbc` and `cell` entries of `hdf_key_mapping` tell the dataset which H5 datasets to read for each molecule.
- `with_pbc: true` tells the collater to add `cell` and `pbc` to every batch. Its default is `false`, in which case the two arrays are read from the file but dropped when the batch is built, and both the prior and the model compute distances without periodic images.

The `with_pbc` flag is only required by `mlcg-train_h5_ng`. The `mlcg-train_h5` script also supports PBC, but there you do not have to specify it: adding the `pbc` and `cell` entries to `hdf_key_mapping` is enough for them to reach the batches, and `with_pbc` is not an option of its data module.

With `cell` and `pbc` in the batch, the prior terms apply the minimum image convention to their distances, and the model builds its neighbor list across the periodic boundaries.

## 2. Noise augmentation

Two options of the `data` section switch on the noise augmentation:

```yaml
data:
  transform_noise_level: 0.001
  transform_kbt: 0.596161081
```

Every time a frame is loaded, `NoiseMapTransformer` adds Gaussian noise to its coordinates and changes its forces to match the noised coordinates:

```
noise       ~ N(0, 1)
sigma       = sqrt(transform_noise_level)
noised_pos  = pos + sigma * noise
noise_force = -kBT * noise / sigma
forces      = noise_force + fmap @ (forces - noise_force)
```

- `transform_noise_level` is the **variance** of the noise, in squared length units of the dataset. Here `0.001` gives `sigma ≈ 0.032`.
- `transform_kbt` is the thermal energy in the energy units of the dataset. Here `0.596161081` kcal/mol corresponds to 300 K.
- `fmap` is the force map stored in `bilayer_nls.pt`. Without an `fmap` entry, the forces are replaced by `noise_force` alone.
- A new noise sample is drawn each time a frame is loaded, so the model sees different coordinates for the same frame in every epoch. The noise is also applied to the validation set.
- The noise level is stored per bead in the batch as `noise_levels`, for models that are conditioned on it.

Setting `transform_noise_level` to `0` or removing it turns the augmentation off.

## 3. Evaluating the prior on the fly

The model is trained on delta forces, i.e. the reference forces minus the prior forces. In the other examples the delta forces are computed once with `mlcg-tk` and stored in the H5 file as `cg_delta_forces`. Here the H5 file holds the **full** forces (`forces: cg_forces` in `hdf_key_mapping`) and the prior is subtracted while each batch is built:

```yaml
data:
  loading_options:
    nl_chkpts:
      POPC: ./bilayer_nls.pt
  baseline_models: ./bilayer_prior.pt
```

- `baseline_models` is the path to the prior. For each batch, the collater moves the data to the GPU, evaluates every prior term on it and subtracts the prior forces from `forces`.
- `nl_chkpts` gives, for each metaset, the file with the neighbor lists that the prior terms need (`bonds` and `angles` here). They are attached to every frame so that they do not have to be rebuilt. After the prior has been subtracted they are removed from the batch (`remove_neighbor_list`, `true` by default).

The on-the-fly evaluation is what makes the noise augmentation possible: the prior forces depend on the coordinates, and the coordinates change with every noise sample, so precomputed delta forces would not match the noised frames. The order inside the collater is:

1. Add the noise to the coordinates and correct the forces (per frame).
2. Collate the frames into a batch, with `cell` and `pbc` if `with_pbc` is set.
3. Evaluate the prior on the noised, periodic batch and subtract its forces.

The loss in `training_config.yaml` (`ForceMSE`) is therefore computed against delta forces, exactly as in the other examples, and the trained model has to be merged with the same prior before simulating:

```bash
mlcg-combine_model --ckpt ./ckpt/last.ckpt --prior ./bilayer_prior.pt --out model_with_prior.pt
```
