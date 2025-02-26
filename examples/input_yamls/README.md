# input yaml examples

This folder contains different input yamls that can be used with the scripts of the `scripts` folder for training a simulation.

## Direct files

The following table descrives each folder and the script to which they can be passed.

| Yaml name | Description | Usage and script | Example |
| :---------: | :---------: | :-------------: | :-------------: |
|`train_schnet.yaml`| Pytorch Lightning Yaml for the training of a traditional CGSchNet model |Training with `./scripts/mlcg-train_h5.py`|`mlcg-train_h5.py fit --config train_schnet_atention.yaml`|
|`train_schnet_attention.yaml`| Pytorch Lightning Yaml for the training of a CGSchNet model with an attention modification  | Training with `./scripts/mlcg-train_h5.py` |`mlcg-train_h5.py fit --config train_schnet_atention.yaml`|
|`train_mace.yaml`| Pytorch Lightning Yaml for the training of a CGMace model | Training with `./scripts/mlcg-train_h5.py` |`mlcg-train_h5.py fit --config train_mace.yaml`|
|`langevin.yaml`|Yaml describing the parameters needed to run a Langevin simulation |Simulating with `./scripts/mlcg-nvt_langevin.py`|`mlcg-nvt_langevin.py --config langevin.yaml`|
|`paralel_tempering.yaml`| Yaml describing the parameters needed to run a parallel tempering simulation |Simulating with `./scripts/mlcg-nvt_pt_langevin.py`|`mlcg-nvt_pt_langevin.py --config parallel_tempering.yaml`|

## Slurm example

The `slurm` folder contains an example of a SLURM bash script, and its accompanying yaml files, for the training of an MLCG model in an HPC cluster managed by [SLURM](https://slurm.schedmd.com/documentation.html)