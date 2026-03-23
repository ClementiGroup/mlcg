#! /usr/bin/env python

import sys
import os.path as osp
from time import ctime
import subprocess
import torch

from mlcg.pl import PLModel, LightningCLI
from mlcg.pl.h5_data_ng import H5DataModule

def main():
    torch.jit.set_fusion_strategy([("DYNAMIC", 3)])
    # to levarage the tensor core if available
    torch.set_float32_matmul_precision("high")

    git = {
        "log": subprocess.getoutput('git log --format="%H" -n 1 -z'),
        "status": subprocess.getoutput("git status -z"),
    }
    print("Start: {}".format(ctime()))

    cli = LightningCLI(
        PLModel,
        H5DataModule,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"error_handler": None},
    )

    print("Finish: {}".format(ctime()))


if __name__ == "__main__":
    main()
