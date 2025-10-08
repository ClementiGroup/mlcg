#! /usr/bin/env python

import sys
import os.path as osp
from time import ctime
import subprocess
import torch
import os

try:
    from lightning.pytorch.plugins.environments import SLURMEnvironment
except (ModuleNotFoundError, ImportError):
    # For PyTorch Lightning <2, this namespace needs to used instead.
    from pytorch_lightning.plugins.environments import SLURMEnvironment


def patch_lightning_slurm_master_addr():
    # Do not patch anything if we're not on a Jülich machine.
    if os.getenv('SYSTEMNAME', '') not in [
            'juwelsbooster',
            'juwels',
            'jurecadc',
            'jusuf',
    ]:
        return

    old_resolver = SLURMEnvironment.resolve_root_node_address

    def new_resolver(*args):
        nodes = args[-1]
        # Append an "i" for communication over InfiniBand.
        return old_resolver(nodes) + 'i'

    SLURMEnvironment.__old_resolve_root_node_address = old_resolver
    SLURMEnvironment.resolve_root_node_address = new_resolver



SCRIPT_DIR = osp.abspath(osp.dirname(__file__))

sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

from mlcg.pl import PLModel, H5DataModule, LightningCLI


if __name__ == "__main__":
    
    patch_lightning_slurm_master_addr()
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
