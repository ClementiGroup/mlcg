#! /usr/bin/env python

from time import ctime
import os.path as osp
import torch
import sys

from mlcg.simulation import (
    parse_simulation_config,
    LangevinConstraint,
)


def main():
    torch.jit.set_fusion_strategy([("DYNAMIC", 3)])
    # to levarage the tensor core if available
    torch.set_float32_matmul_precision("high")

    print(f"Starting simulation at {ctime()} with {LangevinConstraint}")
    (
        model,
        initial_data_list,
        betas,
        simulation,
        profile,
    ) = parse_simulation_config(LangevinConstraint)

    simulation.attach_model_and_configurations(
        model, initial_data_list, beta=betas
    )
    simulation.simulate()
    print(f"Ending simulation at {ctime()}")


if __name__ == "__main__":
    main()