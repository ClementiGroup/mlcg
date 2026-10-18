# Package: mlcg

Machine-Learning Coarse-Graining library. Energy-based models for molecular
systems trained by force matching (forces = `-∇_pos E`), built on PyTorch +
PyTorch Geometric + PyTorch Lightning.

Root: `pkgs/mlcg/src/mlcg/`. When committing changes here: this is its own git repo
(has its own `.git`) — commit inside it first, then update the superproject pointer.

## Top-level layout

| Subpackage | Role |
|------------|------|
| `nn/` | Models, gradient wrappers, losses, priors, radial/angular bases, cutoffs |
| `pl/` | PyTorch-Lightning glue: `PLModel`, `DataModule`, H5 data, `LightningCLI` |
| `data/` | `AtomicData` container + canonical key strings (`_keys.py`) |
| `neighbor_list/` | Neighbor-list backends (torch / nvalchemi / opt_radius) |
| `geometry/` | Distances, angles, dihedrals, topology, statistics, symmetrization |
| `cg/` | Coarse-grain mappings & projection (atomistic → CG beads) |
| `simulation/` | Langevin / parallel-tempering integrators, prior specialization |
| `datasets/` | Chignolin, alanine-dipeptide, octapeptides, H5 (meta)sets |
| `scripts/` | Training / simulation entry points (`mlcg_train*`, `mlcg_nvt_*`) |

## Core data object: `AtomicData` (`data/atomic_data.py`)

Subclasses `torch_geometric.data.Data`. Field names are the `*_KEY` constants in
`data/_keys.py`:

- `pos` `[N,3]`, `atom_types` `[N]`, `n_atoms` `[n_structures]`, `masses`,
  `forces` `[N,3]`, `energy` `[n_structures]`, `velocities`, `cell`, `pbc`,
  `batch` `[N]` (atom→structure map), `neighbor_list` (dict).
- `.out` dict is populated by model forwards, keyed by model `name` then by
  `ENERGY_KEY` / `FORCE_KEY`. `SumOut` also writes top-level `ENERGY_KEY`/`FORCE_KEY`.
- Constructors: `AtomicData.from_ase(frame)`, `AtomicData.from_points(pos, atom_types, ...)`.
- The ctor asserts dtype consistency: `forces`/`energy`/`velocities`/`cell`
  dtype must equal `pos.dtype`. This is the single source of truth for run precision.

## Force-matching mechanics

Forces are never a separate head — they are the analytic negative gradient of
the predicted energy w.r.t. positions, via `torch.autograd.grad`.

`GradientsOut` (`nn/gradients.py`) wraps an energy model:

```python
data.pos.requires_grad_(True)
data = self.model(data)
y = data.out[ENERGY_KEY] (SumOut) or data.out[self.name][ENERGY_KEY]
dy_dr = torch.autograd.grad(y.sum(), data.pos, create_graph=self.training)[0]
data.out[FORCE_KEY] = -dy_dr
```

- `create_graph=self.training` → during training the force graph is retained so the
  loss backward is a 2nd-order derivative (`∂(force-loss)/∂θ`). Inference uses `False`.
- Precision: forces inherit `pos`/parameter dtype exactly. It's exact autodiff
  (no finite-difference / step-size error); only float round-off, which compounds
  because forces are a derivative and training differentiates through them again.
  Default is float32 unless `Trainer(precision="64")` and inputs are cast.
- When configuring a force-matching run, do not enable AMP — the `autograd.grad` runs
  inside autocast and forces get noisy.

Output wrappers (all in `nn/gradients.py`):
- `SumOut(models: ModuleDict, targets)` — sums per-model energies/forces into top-level keys.
- `EnergyOut(model, targets)` — extracts energy only (no gradient).
- `GradientsOut(model, targets=FORCE_KEY)` — adds force via autograd.

**When assembling a model tree, never nest a `SumOut` inside another `SumOut`.**
`SumOut.forward` reads each submodule's output as `data.out[submodule.name][target]` and
accumulates into `data.out[target]`. An inner `SumOut` writes its *children's* keys (and
the top-level `target`), but never `data.out["SumOut"]` (its own default `.name`). So an
outer `SumOut` containing an inner `SumOut` raises `KeyError: 'SumOut'` at forward
(during simulation, not at build time). The merge pipeline
(`merge_priors_and_checkpoint`) already wraps its argument in one `SumOut` — pass it the
bare branch (a model with a real `.name` like `"mace"` / `"kernel_expansion"`), or a flat
`ModuleDict`, never a pre-built `SumOut`. Only build a `SumOut` yourself when summing 2+
sibling branches, and don't then feed that into a prior-merge that wraps again.

## Lightning training loop (`pl/model.py`, `pl/data.py`)

`PLModel(model, loss)`:
- `step()` runs `data = self.model(data)`, copies `data.out[model.name]` to top-level
  `data.out`, then `loss = self.loss(data)`. `derivative` flag auto-true if any
  submodule is a `GradientsOut` (forces grad enabled even in val/test).
- Logs `{stage}_loss`; epochal validation loss is a batch-size-weighted mean across
  metasets (`on_validation_epoch_end`). `on_train_epoch_start` calls `torch.cuda.empty_cache()`.

`DataModule(dataset, log_dir, ...)`: random or file-based splits (`make_splits`),
PyG `DataLoader`s. Batch size `batch_size` (train) / `inference_batch_size` (val/test).
H5-backed alternatives live in `pl/h5_data.py` (metasets).

## Losses (`nn/losses.py`)

`Loss(losses, weights)` sums weighted scalar losses. Components:
- `ForceMSE` / `ForceRMSE` — MSE / sqrt(MSE) on `FORCE_KEY` (the force-matching loss).
- `EnergyMSE` — per-atom energy MSE (`/ n_atoms`).
- `RegL1` — L1 regularization of an arbitrary `data.out` key (e.g. SchNet `radial_filters`).

## Models (`nn/`)

| Model | File | Notes |
|-------|------|-------|
| `SchNet` / `StandardSchNet` | `schnet.py` | CFConv message passing; `StandardSchNet` is the YAML-friendly wrapper |
| `RBFRegularizedSchNet` | `schnet.py` | Per-bead-pair RBF Hadamard weights; emits `radial_filters` for `RegL1` |
| `AttentiveSchNet` / `EdgeRBFRegularizedSchNet` | `schnet.py`, `interpretable_models.py` | Attention-augmented variants |
| `PaiNN`, `MACE`, `So3krates`, `Allegro` | `painn.py`, `mace.py`, `so3krates.py`, `allegro.py` | Equivariant models; `Standard*` wrappers for configs |

SchNet forward: embed types → build/reuse neighbor list → `compute_distances` →
RBF expand → interaction blocks (CFConv) → output MLP → `scatter(..., reduce="sum")`
per structure → `data.out[name] = {ENERGY_KEY: energy}`.

Priors (`nn/prior/`, base `_Prior`): physics terms summed alongside the NN via
`SumOut`. Params stored as buffers (no grad). `Harmonic`, `HarmonicBonds`,
`HarmonicAngles`, `HarmonicImpropers`, `GeneralBonds`, `GeneralAngles`, `Repulsion`
(+ `Cutoff`/`Exp` variants), `FourierSeries`/`Dihedral`, `Polynomial`/`QuarticAngles`,
`RestrictedQuartic`.

Radial bases (`nn/radial_basis/`): `GaussianBasis`, `ExpNormalBasis`, filtered
variants, `RegularizedBasis`. Cutoffs (`nn/cutoff.py`): `CosineCutoff`,
`IdentityCutoff`, `ShiftedCosineCutoff`. When configuring either, give the RBF and the
cutoff the same `cutoff_lower`/`cutoff_upper` (mismatch warns).

## Neighbor lists (`neighbor_list/`)

`atomic_data2neighbor_list(data, rcut, nls_distance_method=...)` → dict via
`make_neighbor_list`. Methods: `"torch"` (default), `"nvalchemi_naive"`,
`"nvalchemi_cell"`, `"nvalchemi_raw"`, `"custom_kernel"`. Optional CUDA fast path
`mlcg_opt_radius.radius_distance` (used by `RBFRegularizedSchNet` when on GPU).
NL dict keys: `tag, order, index_mapping [order,n_edge], cell_shifts, rcut,
self_interaction, mapping_batch`. `validate_neighborlist` checks shape/types.
`order==2` → pairs/distances, `3` → angles, `4` → dihedrals.

## CLI & config (`pl/cli.py`)

`LightningCLI` extends `pytorch_lightning.cli.LightningCLI`:
- Registers torch optimizers + LR schedulers for YAML use.
- `default_root_dir` templating: a literal string `"default_root_dir"` in a
  logger `save_dir`, data `log_dir`, or callback `dirpath` is rewritten to
  `default_root_dir`, `default_root_dir/data`, `default_root_dir/ckpt`
  respectively. **When you see those literals in a YAML, leave them alone** — they are
  intentional, not a bug to be replaced with null or a real path.
- Warns about possible data leakage if `seed_everything` is null/false with
  multi-device training.

## Conventions / gotchas

- When changing run precision: it is governed by `pos.dtype`, and the `AtomicData` ctor
  enforces that all reference targets match it. Cast the inputs and set
  `Trainer(precision=...)` together.
- When touching `SchNet`, leave `self_interaction` at `False` (legacy no-op, raises if `True`).
- When a model consumes `data.pos` upstream of a `GradientsOut`, don't pre-detach it —
  `GradientsOut` sets `requires_grad_(True)` itself and detaches `data.pos` afterwards.
- When editing `geometry/internal_coordinates.py`, keep the `safe_norm` /
  `safe_normalization` calls (eps-guarded) so zero-length vectors don't NaN the backward pass.
