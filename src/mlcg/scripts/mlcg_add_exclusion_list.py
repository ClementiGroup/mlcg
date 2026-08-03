import argparse
import os
from itertools import combinations

import torch


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Command line tool for adding a non-bonded exclusion "
        "list to a set of configurations, computed as the complement of "
        "the existing 'non_bonded' neighbor list within the fully "
        "connected graph. The result is saved next to the input file."
    )
    parser.add_argument(
        "conf_path",
        type=str,
        help="path to the input configurations. Must be a valid .pt file.",
    )
    return parser


def main():
    parser = parse_cli()
    args = parser.parse_args()

    conf_path = args.conf_path
    confs = torch.load(conf_path, weights_only=False)
    for conf in confs:
        actual_nls = conf.neighbor_list["non_bonded"]["index_mapping"]
        fully_connected_nls = torch.tensor(
            list(combinations(range(conf.pos.shape[0]), 2))
        ).T
        num_atoms = conf.pos.shape[0]
        actual_codes = actual_nls[0] * num_atoms + actual_nls[1]
        full_codes = fully_connected_nls[0] * num_atoms + fully_connected_nls[1]
        mask = ~torch.isin(full_codes, actual_codes)
        exclusion_nls = fully_connected_nls[:, mask]
        conf.neighbor_list["non_bonded"][
            "index_mapping_exclusions"
        ] = exclusion_nls

    new_name = conf_path.replace(".pt", "_with_nonbonded_exclusion.pt")
    if not os.path.isfile(new_name):
        print(f"New configurations saved at {new_name}")
        torch.save(confs, new_name)
    else:
        raise ValueError(
            f"File {new_name} exists already, please rename it or move it"
        )


if __name__ == "__main__":
    main()
