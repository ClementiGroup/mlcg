#! /usr/bin/env python

import torch
import sys


from mlcg.nn.gradients import SumOut, NeighborSharingSumOut
from mlcg.pl.utils import (
    extract_model_from_checkpoint,
    merge_priors_and_checkpoint,
)
import argparse


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Command line tool for combining model checkpoints with prior model files, serializing the resulting model as a single PyTorch .pt file."
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to the model checkpoint. Must be a valid .ckpt file.",
    )

    parser.add_argument(
        "--prior",
        default=None,
        type=str,
        help="path to the prior model. Must be a valid .pt file.",
    )
    parser.add_argument(
        "--nls_method",
        type=str,
        default=None,
        choices=[
            "torch",
            "nvalchemi_naive",
            "nvalchemi_cell",
            "nvalchemi_raw",
            "custom_kernel",
        ],
        help="neighbor list method (`nls_distance_method`) to set on the "
        "checkpoint model. If not given, the checkpoint's own setting is kept.",
    )

    parser.add_argument(
        "--convert_to_shared",
        action="store_true",
        help="wrap the combined model in a NeighborSharingSumOut, so the "
        "neighbor list of the checkpoint model is computed once and shared "
        "with the priors listed in --shared_nl_models.",
    )

    parser.add_argument(
        "--shared_nl_models",
        nargs="+",
        default=["non_bonded"],
        help="keys of the prior models that receive the shared neighbor list "
        "when --convert_to_shared is set. Default: non_bonded.",
    )

    parser.add_argument(
        "--out",
        default="combined_model.pt",
        type=str,
        help="directory in which the combined model will be saved.",
    )

    return parser


def main():
    parser = parse_cli()
    args = parser.parse_args()

    if args.prior is not None:
        full_model = merge_priors_and_checkpoint(
            checkpoint=args.ckpt, priors=args.prior
        )
    else:
        full_model = extract_model_from_checkpoint(args.ckpt, None)

    full_model.to("cpu")
    if args.prior is not None:
        # `merge_priors_and_checkpoint` always puts the model from the
        # checkpoint first. We can use that to extract its key
        ckpt_key = next(iter(full_model.models), None)
        if ckpt_key is None:
            raise ValueError("Merged model is empty")
        trained = full_model.models[ckpt_key]
    else:
        trained = full_model
    if args.nls_method is not None:
        checkpoint_model = getattr(trained, "model", trained)
        # setting nls method
        checkpoint_model.nls_distance_method = args.nls_method

    if args.convert_to_shared and args.prior is not None:
        full_model = NeighborSharingSumOut.from_base(
            full_model, ckpt_key, args.shared_nl_models
        )

    torch.save(full_model, args.out)


if __name__ == "__main__":
    main()
