#! /usr/bin/env python

from time import ctime
import torch

from mlcg.simulation import (
    parse_simulation_config,
    REUSSimulation,
)


def main():
    # to levarage the tensor core if available
    torch.set_float32_matmul_precision("high")

    print(f"Starting simulation at {ctime()} with {REUSSimulation}")
    (
        model,
        initial_data_list,
        betas,
        simulation,
        profile,
    ) = parse_simulation_config(REUSSimulation)

    simulation.attach_model_and_configurations(
        model, initial_data_list, beta=betas
    )
    simulation.simulate()
    print(f"Ending simulation at {ctime()}")


if __name__ == "__main__":
    main()
