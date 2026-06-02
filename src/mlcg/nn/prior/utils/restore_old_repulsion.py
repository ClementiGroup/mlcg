from argparse import ArgumentParser
import torch
from pathlib import Path

from mlcg.nn import GradientsOut, SumOut
from mlcg.nn.prior import ExpRepulsion

import sys

sys.path.append(".")

parser = ArgumentParser(
    description="Script to recover priors saved with the old repulsion name"
)

parser.add_argument(
    "--prior",
    type=str,
    help="Path to the prior to correct. Must be a SumOut or module out object of gradients out modules",
)
parser.add_argument(
    "--rep_name",
    type=str,
    default="non_bonded",
    help="Name of the repulsion prior to correct",
)

args = parser.parse_args()
prior = torch.load(args.prior, weights_only=False)
if isinstance(prior, SumOut):
    old_repulsion = prior.models.pop(args.rep_name).model
elif isinstance(prior, torch.nn.ModuleDict):
    old_repulsion = prior.pop(args.rep_name).model
else:
    raise ValueError(
        f"prior at path {args.prior} is not a SumOut or a module dicht"
    )
print(f"Succesfully loaded repulsion at path {args.prior}")
possible_keys = old_repulsion.allowed_interaction_keys
print(f"Found {len(possible_keys)} interaction keys")
stats_dict = {}
for key in possible_keys:
    stats_dict[key] = {
        "alpha": old_repulsion.alpha[key],
        "r_0": old_repulsion.r_0[key],
    }

new_repulsion = ExpRepulsion(stats_dict)
new_repulsion.name = old_repulsion.name
if isinstance(prior, SumOut):
    prior.models[args.rep_name] = GradientsOut(new_repulsion)
elif isinstance(prior, torch.nn.ModuleDict):
    prior[args.rep_name] = GradientsOut(new_repulsion)
print(f"Saving fixed prior")
new_save_name = args.prior.replace(".pt", "_fixed.pt")
if new_save_name == args.prior:
    raise ValueError(
        f"new save name {new_save_name} is the same as the original name, aborting save."
    )
elif Path(new_save_name).is_file():
    raise ValueError(
        f"new save name {new_save_name} already exists, aborting save. Please change the name of existing file and re run."
    )
else:
    torch.save(prior, new_save_name)
