# Input yaml examples

This folder contains different input yamls that can be used with the scripts of the `scripts` folder for training a simulation.

## Direct files

The following table descrives each folder and the script to which they can be passed.

| Yaml name | Description | Usage and script | Example |
| :---------: | :---------: | :-------------: | :-------------: |
|`train_schnet.yaml`| Pytorch Lightning Yaml for the training of a traditional CGSchNet model |Training with `./scripts/mlcg-train_h5`|`mlcg-train_h5 fit --config train_schnet_atention.yaml`|
|`train_schnet_attention.yaml`| Pytorch Lightning Yaml for the training of a CGSchNet model with an attention modification  | Training with `./scripts/mlcg-train_h5` |`mlcg-train_h5 fit --config train_schnet_atention.yaml`|
|`train_mace.yaml`| Pytorch Lightning Yaml for the training of a CGMace model | Training with `./scripts/mlcg-train_h5` |`mlcg-train_h5 fit --config train_mace.yaml`|
|`train_so3krates.yaml`| Pytorch Lightning Yaml for the training of a CGSO3krates model | Training with `./scripts/mlcg-train_h5` |`mlcg-train_h5 fit --config train_so3krates.yaml`|
|`train_painn.yaml`| Pytorch Lightning Yaml for the training of a CGSO3krates model | Training with `./scripts/mlcg-train_h5` |`mlcg-train_h5 fit --config train_painn.yaml`|
|`train_allegro.yaml`| Pytorch Lightning Yaml for the training of a CGAllegro model | Training with `./scripts/mlcg-train_h5` |`mlcg-train_h5 fit --config train_allegro.yaml`|
|`langevin.yaml`|Yaml describing the parameters needed to run a Langevin simulation |Simulating with `./scripts/mlcg-nvt_langevin`|`mlcg-nvt_langevin --config langevin.yaml`|
|`paralel_tempering.yaml`| Yaml describing the parameters needed to run a parallel tempering simulation |Simulating with `./scripts/mlcg-nvt_pt_langevin`|`mlcg-nvt_pt_langevin --config parallel_tempering.yaml`|

## Slurm example

The `slurm` folder contains an example of a SLURM bash script, and its accompanying yaml files, for the training of an MLCG model in an HPC cluster managed by [SLURM](https://slurm.schedmd.com/documentation.html)

## Warning (for developers!)

The tests in `tests/integration/test_examples.py` directly parse and execute the example commands written in this README to validate all the YAML files in the `./input_yamls` folder.  
This means:
- Any modification to the example YAMLs **or** to this document must be done very carefully.  
- Changes may break the automated tests if the commands, file names, or structure no longer match.  

Please review and consider the testing implications before editing this file or the example YAMLs.
