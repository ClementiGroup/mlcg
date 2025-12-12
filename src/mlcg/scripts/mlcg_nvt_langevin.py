#! /usr/bin/env python

from time import ctime
import os.path as osp
import torch
import sys

# Andrea's prior
SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.append(osp.join(SCRIPT_DIR, "../nn/prior/"))

from mlcg.simulation import (
    parse_simulation_config,
    LangevinSimulation,
)


def main():
    torch.jit.set_fusion_strategy([("DYNAMIC", 3)])
    # to levarage the tensor core if available
    torch.set_float32_matmul_precision("high")

    print(f"Starting simulation at {ctime()} with {LangevinSimulation}")
    (
        model,
        initial_data_list,
        betas,
        simulation,
        profile,
    ) = parse_simulation_config(LangevinSimulation)

    simulation.attach_model_and_configurations(
        model, initial_data_list, beta=betas
    )
    simulation.simulate()
    print(f"Ending simulation at {ctime()}")


if __name__ == "__main__":
    main()
