#! /usr/bin/env python

from time import ctime
import os.path as osp
import torch
import sys
from typing import Any, Dict


from mlcg.simulation import (
    parse_simulation_config,
    PTSimulation,
)

def main():
    print(f"Starting simulation at {ctime()} with {PTSimulation}")
    (
        model,
        initial_data_list,
        betas,
        simulation,
        profile,
    ) = parse_simulation_config(PTSimulation)

    simulation.attach_model_and_configurations(
        model, initial_data_list, betas=betas
    )
    simulation.simulate()
    print(f"Ending simulation at {ctime()}")



if __name__ == "__main__":
    main()