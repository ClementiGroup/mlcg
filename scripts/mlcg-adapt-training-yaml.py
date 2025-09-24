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
        # remove unsupported args
        trainer_section.pop("track_grad_norm")
        trainer_section.pop("auto_lr_find")
        assert "track_grad_norm" not in old_yaml["trainer"].keys()
        # change callbacks name
        if "callbacks" in trainer_section.keys():
            for callback_dict in trainer_section["callbacks"]:
                if callback_dict["class_path"]=="pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint":
                    callback_dict["class_path"]="mlcg.pl.OffsetCheckpoint"
                    if "start_epoch" not in callback_dict["init_args"]:
                        callback_dict["init_args"]["start_epoch"] = 0
        # change location of resuming checkpoint argument
        if "resume_from_checkpoint" in trainer_section.keys():
            old_ckpt = trainer_section.pop("resume_from_checkpoint")
            old_yaml["ckpt_path"] = old_ckpt
        dump_yaml(args.out,old_yaml)