# Single molecule example

This readme contains the instructions to train a model in a dataset with data of a single molecule, protein [1L2Y](https://www.rcsb.org/structure/1L2Y), a mutant of TrpCage.

All of the files conained in this folder are analogous to the ones generated in the [mlcg-tk example folder](https://github.com/ClementiGroup/mlcg-tk/tree/main/examples).

## Step 1: Training a model

We will train a model with the `1L2Y_prior_tag.h5` dataset that was previously generated using `mlcg-tk`. To train this model, we need the partition file that specifies how to split the dataset into a training and validation region: file `partition_1L2Y_prior_tag.yaml`. These two files define the dataset, its partition and its batch size that will be used to train a model

To go ahead to the trainng, the file `training.yaml` is a Pytorch lightning yaml defines the architecture of the model to use (`model` field), the optimizer (`optimizer`), the trainer specifications (`trainer`) and the dataset (`dataset`). Note that `data.h5_file_path` and `data.partition_options` point to the `1L2Y_prior_tag.h5` and `partition_1L2Y_prior_tag.yaml` files, respectively, that will be used as training data and 

To train, we can run from a terminal (that)

```bash
mlcg-train_h5.py fit --config ./training.yaml
```

A progress bar will appear showing how much time per epoch the model can take. Note that the progress bar can be disabled by setting `trainer.enable_progress_bar: False` in the training yaml.

During training, the ouput will be saved into two directories: 
- `${trainer.default_root_dir}/ckpt` will save the state of the model every certain epoch. This checkpoints can be examined with pytorch. By default the last checkpoint is saved in `last.ckpt`
- `${trainer.default_root_dir}/tensorboard` will save a tensorboard logger that can be used to launch a [tensorboard](https://www.tensorflow.org/tensorboard) server to monitorize the training and validation loss of the model (i.e. running `tensorboard --logdir ${trainer.default_root_dir}/tensorboard`)

## Step 2: Joining a certain epoch with a prior object to simulate

After the model has been trained for several epochs, we can merge it with a prior object in order to be able to simulate it.
For this, we need to use the `mlcg-combine_model.py` script and pass as arguments the desired checkpoint, the prior object and 
a path for saving the merged object:

```bash
mlcg-combine_model.py --ckpt ./ckpt/last.ckpt --prior ./prior_tag_prior_model.pt --out model_with_prior.pt
```

This command might throw some warnings related to a rank problem but this are safe to ignore. 

## Step 3: Simulating the model 

With our model merged with the prior, it is now possible to run a simulation of this model. To ensure proper file management, we first create a folder to store the simulation output:

```bash
mkdir sims
```

After this, we can use the script `mlcg-nvt_langevin.py` to run a simulation from the terminal. 
The simulation parameters and configuration are specified in `1L2Y_sim_demo.yaml`. The field `model_file` points to our merged model with a prior, and the `structure_file` points 
to `1L2Y_model_tag_configurations.pt` file which stores the initial configurations of the 35 
1L2Y structures that we will simulate.

To run the simulation we can just use:

```bash
mlcg-nvt_langevin.py --config ./1L2Y_sim_demo.yaml
```

A progress bar will appear detailing the status of our simulation.

### Warning: possible problems with simulation

In the 1L2Y example, it is possible that the simulation exits before finishing after trowing an error related to "Simulation blewup at timestep #..."

This problem is related to the fact that the prior was fitted with very little data and is not a good enough prior to avoid unphysical configurations in our system. 

## Step 4: Analyzing a simulation. 

In the `sims` folder, we will see some files:
- `1L2Y_log.txt`, a log file 
- `1L2Y_config.yaml`, an expanded version of the simulation yaml with all the relevant parameters.
- `1L2Y_specialized_model_and_config.pt` which has the network model and the starting configurations
- `1L2Y_coords_????.npy`, numpy arrays detailing the coordinates of the simulation.

To see how to load the `sims/1L2Y_coords_????.npy` numpy coordinates and convert them into an mdtraj.Trajectory 
using the `1L2Y_cg_structure.pdb` file which defines the topology of the system, 
please check the `load_trajectories.ipynb` notebook. 