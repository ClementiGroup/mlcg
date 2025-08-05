import sys
import os.path as osp
from time import ctime
import subprocess
import torch

SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))
sys.path.insert(0, "/srv/data/kamenrur95/mlcg/")  # Add MLCG path

from mlcg.pl import H5DataModule, LightningCLI
from mlcg.nn.standard_allegro import StandardAllegro  # Import your class

if __name__ == "__main__":
    torch.jit.set_fusion_strategy([("DYNAMIC", 3)])
    torch.set_float32_matmul_precision("high")

    git = {
        "log": subprocess.getoutput('git log --format="%H" -n 1 -z'),
        "status": subprocess.getoutput("git status -z"),
    }
    print("Start: {}".format(ctime()))

    # Use your StandardAllegro instead of PLModel
    cli = LightningCLI(
        StandardAllegro,        # Your Allegro model
        H5DataModule,           # Keep the H5 data module
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"error_handler": None},
    )

    print("Finish: {}".format(ctime()))