#! /usr/bin/env python

from time import ctime
import os.path as osp
import torch
import sys

# Andrea's prior
SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.append(osp.join(SCRIPT_DIR, "../nn/prior/"))

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
        type=str,
        help="path to the prior model. Must be a valid .pt file.",
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

    full_model = merge_priors_and_checkpoint(
        checkpoint=args.ckpt, priors=args.prior
    )
    full_model.to("cpu")
    torch.save(full_model, args.out)


if __name__ == "__main__":
    main()
