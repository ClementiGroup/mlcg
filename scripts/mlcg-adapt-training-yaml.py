from mlcg.utils import load_yaml, dump_yaml
import argparse

def parse_cli():
    parser = argparse.ArgumentParser(
        description="Command line tool for adapting training yamls from lightning version 1.9.4 to 2.2.1"
    )
    parser.add_argument(
        "--training_yaml",
        type=str,
        help="path to the training yaml. It must contain a `trainer` field.",
    )

    parser.add_argument(
        "--out",
        default="lightning_2-2-1_training.yaml",
        type=str,
        help="path for the new yaml to be saved.",
    )

    return parser

_keys_to_remove = [
    "multiple_trainloader_mode",
    "move_metrics_to_cpu",
    "amp_backend",
    "amp_level",
    "auto_scale_batch_size",
    "replace_sampler_ddp",
    "auto_lr_find",
    "ipus",
    "tpu_cores",
    "auto_select_gpus",
    "gpus",
    "num_processes"
]

if __name__ == "__main__":
    parser = parse_cli()
    args = parser.parse_args() 
    old_yaml = load_yaml(args.training_yaml)
    if "trainer" not in old_yaml.keys():
        raise ValueError(
            "Provided yaml doesn't contain a `trainer` section."
        )
    else:
        trainer_section = old_yaml["trainer"]

        # adapt grad norm tracking
        track_grad_norm = trainer_section.pop("track_grad_norm", -1)
        if track_grad_norm != -1:
            print(
                "Grad norm tracking is not directly supported in lightning 2.2.1. "
                "You can instead add an mlcg.pl.GradNormLogger callback to your callbacks list."
            )

        assert "track_grad_norm" not in old_yaml["trainer"].keys()
        
        # remove unsupported args
        for key in _keys_to_remove:
            trainer_section.pop(key, None)
            assert key not in old_yaml["trainer"].keys()

        # change location of resuming checkpoint argument
        if "resume_from_checkpoint" in trainer_section.keys():
            old_ckpt = trainer_section.pop("resume_from_checkpoint")
            old_yaml["ckpt_path"] = old_ckpt
        dump_yaml(args.out,old_yaml)