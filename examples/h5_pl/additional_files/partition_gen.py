import sys
import numpy, h5py
from mlcg.datasets.split_utils import multimol_split, n_fold_multimol_split
import argparse
from mlcg.utils import load_yaml, dump_yaml
from ruamel.yaml import YAML

yaml = YAML(pure="true", typ="safe")
yaml.default_flow_style = False

yaml = YAML()


def prepare_molecule_dictionary(h5_file):
    f = h5py.File(h5_file)

    mol_dicts = {}
    for name in f.keys():
        mol_dict = {}
        for k in f[name].keys():
            mol_dict[k] = f[name][k].attrs["N_frames"]
        mol_dicts[name] = mol_dict
    return mol_dicts


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Tool for splitting pre-generated H5 datasets with potentially multiple molecules. "
        "See mlcg.datasets.H5Dataset for more information. If you have not generated an h5 "
        "file from NumPy files yet, please use save_h5.py. You can also generate partitions "
        "using explore_h5_dataset.ipynb."
    )
    parser.add_argument(
        "--h5path",
        type=str,
        help="path to the h5 dataset",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./my_partions.yaml",
        help="filename where the splits will be saved in YAML format",
    )
    parser.add_argument(
        "--kfsplits",
        default=None,
        type=int,
        help="If specified, the number of folds to split the dataset into. Otherwise, a single split will be made.",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="If specified, the splitting or KFfold splitting will be seeded",
    )
    parser.add_argument(
        "--proportions",
        nargs="+",
        default=[0.8, 0.2],
        type=list,
        help="For non-KFold splitting, this list gives [train_proportion, validation_proportion]",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        type=bool,
        help="Display additional information about splits",
    )

    return parser


if __name__ == "__main__":
    parser = parse_cli()
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        exit()

    mol_dicts = prepare_molecule_dictionary(args.h5path)
    verbose = args.verbose

    output = {}
    for name in mol_dicts.keys():
        mol_dict = mol_dicts[name]
        if args.kfsplits != None:
            splits = n_fold_multimol_split(
                mol_dict, k=args.kfsplits, shuffle=True, random_state=args.seed
            )
        else:
            if len(args.proportions) != 2:
                raise ValueError(
                    "Proportions must be two entries: train and validation."
                )
            props = {"train": args.proportions[0], "val": args.proportions[1]}
            splits = multimol_split(
                mol_dict,
                proportions=props,
                random_seed=args.seed,
                verbose=verbose,
            )
        output[name] = splits
    dump_yaml(args.out, output)
