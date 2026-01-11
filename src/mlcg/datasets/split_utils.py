from typing import Dict, Union, List
import numpy as np
import warnings
from sklearn.model_selection import KFold


__all__ = ["mol_split", "multimol_split"]
"""
Train-Validation splitting strategy
No cross validation concerns for the moment
we process everything in the sorted key order, so as to keep consistency over different runs/processes/machines
For each Metaset
  - if it only contains single molecule, then split the underlying data into train-val sets according to the ratio
  - if it contains multiple molecules, then split on the level of molecules,
    (so that one molecule can only be in *either* train or val set, and the *total number of samples* is roughly according to the desired split)
"""


def _check_props(part_props):
    """Checking whether the provided ratio is valid."""
    props = np.array(list(part_props.values()))
    # all entries should be non-negative
    assert np.all(props >= 0.0), "all proportions should be non-negative"
    # it shall not add up more than one (considering numerical error)
    assert np.sum(props) < 1 + 1e-4, "proportions add up more than 1."


# Method 1: splitting single-molecule metaset
# get n_frames from the single-molecule data, establish an index, randomly shuffle it (with a given random seed), and then
# split the molecular data accordingly


def mol_split(
    n_frames,
    proportions={"train": 0.8, "val": 0.2},
    random_seed=42,
    verbose=False,
):
    _check_props(proportions)
    indices = np.arange(n_frames)
    rng = np.random.default_rng(random_seed)
    rng.shuffle(indices)
    split_indices = {}
    start_id = 0
    for part_name, part_prop in proportions.items():
        target_id = start_id + int(n_frames * part_prop)
        target_id = min(target_id, n_frames)
        split_indices[part_name] = indices[start_id:target_id]
        split_indices[part_name].sort()
        start_id = target_id
    if verbose:
        msg = "Actual split:\n"
        for part_name, part_indices in split_indices.items():
            msg += '- "%s": %d (%.1f%%)\n' % (
                part_name,
                len(part_indices),
                len(part_indices) / n_frames * 100.0,
            )
        print(msg)
    return split_indices


# Method 2: splitting multi-molecule metaset
# we consider all frames belong to each molecule as a whole in the split
# get n_frames from the single-molecule data, establish an index,
# randomly shuffle it (with a given random seed), and then split the molecular data accordingly


def multimol_split(
    mol_count_dict: Dict[str, int],
    proportions: Dict[str, float] = {"train": 0.8, "val": 0.2},
    random_seed: int = 42,
    train_names: Union[np.ndarray, None] = None,
    val_names: Union[np.ndarray, None] = None,
    verbose: bool = False,
) -> Dict[str, List[str]]:
    r"""Function for splitting molecules into training and validation sets. The assignment
    of frames to either the train or validation set is done molecule-wise. That is, a given
    molecule's data is either all in the train set, or all in the validation set. The partitioning
    of molecules into either the train or test set is done such that the proportion of total train
    frames against the total validation frames matches the supplied train:validation propportions.
    The user can override this by specifically passing both `train_names` and `val_names` for their
    own preferred molecular split.

    Parameters
    ----------
    mol_count_dict:
        Dictionary mapping molecule names to their total simulation frame counts
    proportions:
        Dictionary mapping 'train' and 'val' sets to their proportions of the full set.
        E.g.) a 5:1 train:test split would be given as:

        .. code::

            proportions = {'train': 0.8, 'val': 0.2}

        Proportions are taken molecule-wise. If `train_names` and `val_names` are not specified,
        then the molecule splits are made such that the number of train frames and validation frames
        also match the desired the proportion. Users should check to see if their molecular splits
        make sense if they have molecules with vastly different numbers of frames.

    random_seed:
        Seeds the shuffling of molecule names prior to splitting.

    train_names:
        Numpy array of strings containing the exact names of the molecules desired
        for the train set. If not None, val_names must also be specified. No shuffling
        is done.
    val_names:
        Numpy array of strings containing the exact names of the molecules desired
        for the validation set. If not None, train_names must also be specified.
        No shuffling is done.
    verbose:
        If True, information about the splits will be given when both train_names
        and val_names are None

    Returns
    -------
    split_mols:
        Dictionary of strings 'train' and 'val' mapping to lists of molecule names
        for the train and validation sets respectively.
    """

    is_all_none = all(
        [isinstance(opt, type(None)) for opt in [train_names, val_names]]
    )
    is_all_numpy = all(
        [isinstance(opt, np.ndarray) for opt in [train_names, val_names]]
    )

    if not is_all_none and not is_all_numpy:
        raise ValueError(
            "train_names = {} but val_names = {}. They must be both None or both np.ndarray".format(
                train_names, val_names
            )
        )

    if is_all_none:
        _check_props(proportions)
        mol_names = np.array(sorted(mol_count_dict))
        mol_frames = np.array([mol_count_dict[n] for n in mol_names])
        indices = np.arange(len(mol_names))
        rng = np.random.default_rng(random_seed)
        rng.shuffle(indices)
        # we go for a split whose train_ratio is closest to the desired value
        accum_frames = np.cumsum(mol_frames[indices])
        n_frames = accum_frames[-1]
        split_parts = {}

        def get_part_prop(part_name, count=False):
            if not part_name in split_parts:
                return 0.0
            lb, ub = split_parts[part_name]
            return get_range_prop(lb, ub, count)

        def get_range_prop(lb, ub, count=False):
            frame_start = 0 if lb == 0 else accum_frames[lb - 1]
            frame_stop = 0 if ub == 0 else accum_frames[ub - 1]
            n_frames_part = frame_stop - frame_start
            if count:
                return n_frames_part
            else:
                return n_frames_part / n_frames

        split_prop = 0.0
        last_split_index = 0
        for part, prop in sorted(
            list(proportions.items()), key=lambda tup: -tup[1]
        ):
            target_prop = split_prop + prop
            split_index = (
                np.argmin(np.abs(accum_frames - n_frames * target_prop)) + 1
            )
            split_parts[part] = (last_split_index, split_index)
            split_prop += get_range_prop(last_split_index, split_index)
            last_split_index = split_index

        if any((get_part_prop(p) == 0.0 for p in proportions)):
            warnings.warn(
                "No solution with a partition ratio close to demanded is found:"
                " at least one partition is empty."
            )
        # transcribe the results
        split_mols = {}
        for part in split_parts:
            lb, ub = split_parts[part]
            split_mols[part] = [mol_names[i] for i in sorted(indices[lb:ub])]
        if verbose:
            msg = "Actual split:\n"
            n_frames = accum_frames[-1]
            for part_name, part_range in split_parts.items():
                n_frames_part = get_part_prop(part_name, count=True)
                msg += f"- `{part_name}`: {n_frames_part} ({n_frames_part / n_frames * 100.:.1f}%)\n"
            print(msg)
        split_mols["train"] = _sanitize_strings(split_mols["train"])
        split_mols["val"] = _sanitize_strings(split_mols["val"])
        return split_mols

    if is_all_numpy:
        split_mols = {}
        mol_names = np.array(sorted(mol_count_dict))

        # check to make sure there are no unknown molecules not found
        # in mol_names
        train_mol_check = [name in mol_names for name in train_names]
        val_mol_check = [name in mol_names for name in val_names]
        if not all(train_mol_check):
            unkown_idx = [not el for el in train_mol_chec]
            unknown_mols = train_names[unknown_idx]
            raise ValueError(
                "Unknown molecules {} in train_names".format(unkown_mols)
            )
        if not all(val_mol_check):
            unkown_idx = [not el for el in val_mol_chec]
            unknown_mols = train_names[unknown_idx]
            raise ValueError(
                "Unknown molecules {} in val_names".format(unkown_mols)
            )

        # Warnings about leaky sets and set coverage
        if len(set(train_names).intersection(val_names)) != 0:
            warnings.warn(
                "Overlap detected between train and val molecule names."
            )
        if len(train_names) + len(val_names) != len(mol_names):
            warnings.warn(
                "Split portions do not cover the entire set of molecule names."
            )

        split_mols["train"] = _sanitize_strings(train_names)
        split_mols["val"] = _sanitize_strings(val_names)
        return split_mols


def n_fold_multimol_split(
    mol_count_dict: Dict[str, int],
    k: int = 5,
    shuffle: bool = True,
    random_state: Union[None, int] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """Function for creating k non-overlapping molecule train/validation sets
    for use in cross validation experiments.

    Parameters
    ----------

    mol_count_dict:
        Dictionary mapping molecule names to their total simulation frame counts
    k:
        Number of cross validation folds
    shuffle:
        If True, the molecule names will be shuffled prior to splitting each fold
    random_state:
        If not None, seeds the fold splitting process

    Returns
    -------

    k_fold_splits:
        Dictionary of splits for each fold
    """

    mol_names = np.array(sorted(mol_count_dict))
    folder = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
    k_fold_splits = {}
    for i, (train_idx, val_idx) in enumerate(folder.split(mol_names)):
        train_names = mol_names[train_idx]
        val_names = mol_names[val_idx]
        k_fold_splits["fold_{}".format(i)] = multimol_split(
            mol_count_dict, train_names=train_names, val_names=val_names
        )
    return k_fold_splits


def _sanitize_strings(array: np.ndarray) -> List[str]:
    """Helper function to santize numpy arrays of strings
    for YAML representation

    Parameters
    ----------
    array:
        Numpy array of strings

    Returns
    -------
        List of strings
    """
    sanitized = [str(element) for element in array]
    return sanitized
