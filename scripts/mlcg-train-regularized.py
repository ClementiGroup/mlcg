#! /usr/bin/env python

import sys
import os.path as osp
from time import ctime
import subprocess
import torch

SCRIPT_DIR = osp.abspath(osp.dirname(__file__))

sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

from mlcg.pl import RegularizedPLModel, DataModule, LightningCLI


if __name__ == "__main__":
    torch.jit.set_fusion_strategy([("DYNAMIC", 3)])
    # to levarage the tensor core if available
    torch.set_float32_matmul_precision("high")
    git = {
        "log": subprocess.getoutput('git log --format="%H" -n 1 -z'),
        "status": subprocess.getoutput("git status -z"),
    }
    print("Start: {}".format(ctime()))

    cli = LightningCLI(
        RegularizedPLModel,
        DataModule,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"error_handler": None},
        auto_configure_optimizers=False,
    )

    print("Finish: {}".format(ctime()))
