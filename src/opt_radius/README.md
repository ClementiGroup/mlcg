# mlcg_opt_radius

This submodule provides a custom CUDA kernel for neighbor list construction 
that can be used by SchNet. In most cases this kernel has a faster performance
than the naive python implementation.

This kernel also provides additional features such as the possibility of excluding
certain edges from the graph defined by the neighbohr list.

## Pre-installation: NVCC match

Contrary to `mlcg`, which can be ran just using the CUDA runtime, this module
compiles the custom kernel and as such it requires to have an existing installation 
of nvcc, the CUDA compiler.

Pytorch will only be able to use this kernel if the installed Pytorch version is 
compatible with the major version of NVCC (for example, PyTorch for cuda 12.1 will 
work )

**Make sure that your PyTorch installation is compatible with NVCC**. You can find 
the instruction to install PyTorch for a variety of CUDA versions at their [official documentation](https://pytorch.org/get-started/locally/)


## Installation 

From this directory, run `python setup.py develop`. 

Some information from the NVCC will print on the screen. After some minutes, installation should be complete.

## Usage

If installed, the custom kernel will be used by default without further tweaking.

### Pair exclusion

To enable the exclusion of certain edges from the neighbohrlist, you need to pass
the exclusion list as an attribute of the molecule in an H5 dataset. See the 
description of the structure in  documentation of `H5Dataset` for further idea.

An example of the training input yaml enabling the exclusions  can be found in `examples/input_yamls/train_schnet_exclusions.yaml`