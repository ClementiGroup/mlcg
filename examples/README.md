# Examples


This folder contains different examples on how to use this package for traning and simulating an MLCG model. The examples are organized in folders. 

## Disclaimer

**These examples are intented to just be illustrative on how to use this code but they are not supposed to produce any meaningful, useful or "MD-production-ready" MLCG model**

We only provide this examples to show the user how to use the codebase. Training an MLCG model usually requires a significant amount of data which could not be saved on web-hosted repository.  

## Contents

The following table describes each folder and their contents

| Folder name | Content description | Target audience |
| :---------: | :---------: | :-------------: |
|`notebooks`|Notebook showing the training, simulation and simulation analysis of an MLCG model for Chignolin | People interested in understanding the procedure for building and testing an mlcg model but not interested in getting a good model nor applying it to a system of their own. |
|`h5_pl`| Files and input yamls for training a toy model of 1L2Y and a transferable model  | People who intend to build an mlcg model to a system of their own and that have gone through the [mlcg-tk package example](https://github.com/ClementiGroup/mlcg-tk/tree/main/examples) for preparing AA data into a trainable H5 |
| `input_yamls`| Example yaml files that can be passed to the scripts  | People that went trough the examples in `h5_pl` folder  |

