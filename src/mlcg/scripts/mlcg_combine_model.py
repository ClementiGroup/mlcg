#! /usr/bin/env python

import torch
from mlcg.nn.kernels.converter import convert_standard_model_to_flash

from mlcg.nn.gradients import SumOut
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
        "--convert_to_flash",
        action="store_true",
        help="if set, the script will attempt to convert the loaded model to its flash counterpart",
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

    if args.convert_to_flash:
        full_model = convert_standard_model_to_flash(full_model)

    full_model.to("cpu")
    torch.save(full_model, args.out)


if __name__ == "__main__":
    main()
