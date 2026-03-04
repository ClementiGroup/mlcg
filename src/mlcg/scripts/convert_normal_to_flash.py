from argparse import ArgumentParser
import importlib
import torch
import os

from mlcg.nn import (
    SumOut,
    GradientsOut,
    StandardFlashSchNet,
    load_and_adapt_old_checkpoint,
)
from mlcg.pl.utils import merge_priors_and_checkpoint
from mlcg.utils import load_yaml


def get_class_init_and_init_args(class_dict):
    init_args = class_dict.pop("init_args", {})
    splitted = class_dict["class_path"].split(".")
    loc = ".".join(splitted[:-1])
    module = importlib.import_module(loc)
    curr_class = getattr(module, splitted[-1])
    return curr_class, init_args


def parse_cli():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--state_dict", type=str)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--prior", type=str)
    parser.add_argument("--out", type=str, default="flashschnet.pt")
    return parser.parse_args()


def main():
    args = parse_cli()
    config_yaml = load_yaml(args.config)

    schnet_args = config_yaml["model"]["model"]["init_args"]["model"][
        "init_args"
    ]

    ind_components = [
        "activation",
        "cutoff",
        "rbf_layer",
    ]
    for com in ind_components:
        class_dict = schnet_args.pop(com)
        init, init_args = get_class_init_and_init_args(class_dict)
        new_class = init(**init_args)
        schnet_args[com] = new_class

    init, init_args = get_class_init_and_init_args(
        config_yaml["model"]["model"]["init_args"]["model"]
    )

    new_net = StandardFlashSchNet(**init_args)

    if os.path.isfile(args.ckpt):
        print("Trying to load checkpoint")
        checkpoint = load_and_adapt_old_checkpoint(args.ckpt)
        state_dict = checkpoint["state_dict"]
    elif os.path.isfile(args.state_dict):
        print("Trying to load state dictionary")
        state_dict = torch.load(args.state_dict)
    else:
        raise ValueError(
            "Not possible to run without providing a valid checkpoint or state dict"
        )

    clean_state_dict = {
        k.replace("model.model.", ""): v
        for k, v in state_dict.items()
        if k != "loss.weights"
    }
    new_net.load_state_dict(clean_state_dict)

    grad_net = GradientsOut(new_net, targets="forces")

    prior = torch.load(args.prior)
    if isinstance(prior, SumOut):
        prior = prior.models

    full_net = merge_priors_and_checkpoint(grad_net, prior)

    torch.save(full_net, args.out)
    return 0


# print(checkpoint.keys())

if __name__ == "__main__":
    main()
