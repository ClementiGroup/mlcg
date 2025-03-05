# Additional files for H5 datasets

This folder contains files that can be useful when working with H5 datasets for building MLCG models

## Yaml files

| File name | Description | Usage and script | Example |
| :---------: | :---------: | :-------------: | :-------------: |
|`train_h5_1_10.yaml`| Pytorch Lightning yaml for the training of a traditional CGSchNet model using an H5 dataset | Training with `./scripts/mlcg-train_h5.py`|`mlcg-train_h5.py fit --config train_h5_1_10.yaml`|
|`train_h5_1_10_exclusion.yaml`| Pytorch Lightning yaml for the training of a CGSchNet model with bond-exclusion using an H5 dataset | Training with `./scripts/mlcg-train_h5.py`|`mlcg-train_h5.py fit --config train_h5_1_10_exclusion.yaml`|
|`partition_settings.yaml`| yaml describing how to partition an H5 dataset into training and testing part | Specifying the partition in a training yaml that will be passed to `./scripts/mlcg-train_h5.py` |`mlcg-train_h5.py fit --config train_h5_1_10_exclusion.yaml --data.partition_options partition_options.yaml`|

## Python scripts and notebooks

| File name | Description | Usage | 
| :---------: | :---------: | :-------------: | 
|`explore_h5_dataset.ipynb`| Notebook that shows how to open an h5 dataset and see its content,and also how to generate a partition from it | Exploring an H5 dataset and generating a partition yaml |
|`partition_gen.py`| Script for generating a partition yaml from an existing h5 dataset. The argument parser provides the necessary information for how to run it | Generating a partition yaml |
|`pdb_str_util.py`| Functions to serialize a PDB file into a string so that it can be saved as an attribute in an H5 dataset | Serializing PDB files in an H5 dataset |
