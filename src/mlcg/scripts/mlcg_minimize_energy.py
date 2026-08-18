#! /usr/bin/env python

from time import ctime
import torch

from mlcg.simulation import parse_minimizer_config, minimize_energy


def main():
    print(f"Starting minimization at {ctime()} with {minimize_energy}")
    (
        model,
        initial_data_list,
        minimizer_kwargs,
        output_file,
    ) = parse_minimizer_config()

    minimized = minimize_energy(model, initial_data_list, **minimizer_kwargs)
    torch.save(minimized, output_file)
    print(f"Ending minimization at {ctime()}")


if __name__ == "__main__":
    main()
