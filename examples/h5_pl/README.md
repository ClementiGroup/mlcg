# Example for training using H5 dataset and Pytorch Lightning model specification

This example shows how to train an MLCG model by using an H5 dataset and Pytorch Lighning 

- Bundled datasets in single (or several) HDF5 files.
- Parallelized training on multiple GPUs with distributed data parallel (DDP) and low memory footprint.
- External description of dataset partition that allows a simple way of defining the training and validation splits. 
- Balanced batch sized with accurate data loading proportions.

## H5 dataset construction: the `mlcg-tk` package

We provide the tools to convert an AA dataset into an H5 dataset compatible with mlcg in the [mlcg-tk package](https://github.com/ClementiGroup/mlcg-tk/). Please go to the [mlcg-tk example folder](https://github.com/ClementiGroup/mlcg-tk/tree/main/examples) folder for a detailed description on how to use mlcg-tk.

## Single protein model example

Check the folder `single_molecule` to learn how to make an mlcg model for an H5 dataset that contains data from a single molecule. 

## Multi molecule example

Check the folder `multiple_molecules` to see how to train an mlcg on an H5 dataset that contains data from multiple molecules.

## Additional files

In the `additional_files` folder you can find some useful yaml input examples for mlcg scripts and some code that can be useful when working with H5 datasets.

## Note

1. The detailed data structure information of an H5 Dataset can be found at the [documentation](https://clementigroup.github.io/mlcg/)
2. The pytorch-lightning script `../../scripts/mlcg-train_h5.py` may serve as an example for constructing training pipelines without pytorch-lightning.
