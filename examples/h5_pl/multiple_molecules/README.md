# Multi molecule molecule example

This readme contains the instructions to train a model in a dataset with several molecules

## Step 1: Training a model

We will train a model with the `combined_demo_dataset.h5` dataset. To train this model, we need the partition file that specifies how to split the dataset into a training and validation region: file `partition_demo.yaml`. These two files define the dataset, its partition and its batch size that will be used to train a model

To go ahead to the trainng, the file `train_demo_cuda.yaml` is a Pytorch Lightning yaml defines the architecture of the model to use (`model` field), the optimizer (`optimizer`), the trainer specifications (`trainer`) and the dataset (`dataset`). Note that `data.h5_file_path` and `data.partition_options` point to the `combined_demo_dataset.h5` and `partition_demo.yaml` files, respectively, that will be used as training data and 

To train, we can run from a terminal:

```bash
mlcg-train_h5.py fit --config ./train_demo_cuda.yaml
```

During training, the ouput will be saved into two directories: 
- `${trainer.default_root_dir}/ckpt` will save the state of the model every certain epoch. This checkpoints can be examined with pytorch. By default the last checkpoint is saved in `last.ckpt`
- `${trainer.default_root_dir}/tensorboard` will save a tensorboard logger that can be used to launch a [tensorboard](https://www.tensorflow.org/tensorboard) server to monitorize the training and validation loss of the model (i.e. running `tensorboard --logdir ${trainer.default_root_dir}/tensorboard`)

## Step 2: Joining a certain epoch with a prior object to simulate

After the model has been trained for several epochs, we can merge it with a prior object in order to be able to simulate it.
For this, we need to use the `mlcg-combine_model.py` script and pass as arguments the desired checkpoint, the prior object and 
a path for saving the merged object:

```bash
mlcg-combine_model.py --ckpt ./ckpt/last.ckpt --prior ./prior.pt --out model_with_prior.pt
```

This command might throw some warnings related to a rank problem but these are safe to ignore. 

## Step 3: Simulating the model 

With our model merged with the prior, it is now possible to run a simulation of this model. To ensure proper file management, we first create a folder to store the simulation output:

```bash
mkdir sims
```

After this, we can use the script `mlcg-nvt_langevin.py` to run a simulation from the terminal. 
The simulation parameters and configuration are specified in `cln_sim_demo.yaml`. The field `model_file` points to our merged model with a prior, and the `structure_file` points 
to `cln_configurations_demo.pt` file which stores the initial configurations of the 10 
CLN structures that we will simulate.

To run the simulation we can just use:

```bash
mlcg-nvt_langevin.py --config ./cln_sim_demo.yaml
```

A progress bar will appear detailing the status of our simulation.

## Step 4: Analyzing a simulation. 

In the `sims` folder, we will see some files:
- `cln_log.txt`, a log file 
- `cln_config.yaml`, an expanded version of the simulation yaml with all the relevant parameters.
- `cln_specialized_model_and_config.pt` which has the network model and the starting configurations
- `cln_coords_????.npy`, numpy arrays detailing the coordinates of the simulation.

The `cln_coords_????.npy` contain the chunks of simulations as defined by `export_interval` in the simulation yaml. Each numpy file contains an array
of shape `(n_trajectories, n_frames_per_chunk, n_cg_beads, 3)`, where `n_frames_per_chunk = export_interval//save_interval` and `n_trajectories` corresponds to the number of configurations in your initial simulation input `cln_configurations_demo.pt`.
To see how to load the `sims/cln_coords_????.npy` numpy coordinates and convert them into an mdtraj.Trajectory using the `cln_5beads.pdb` file, please check the `load_trajectories.ipynb` notebook in the `../single_molecule` folder.