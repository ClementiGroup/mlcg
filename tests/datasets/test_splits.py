import pytest
import numpy as np
from mlcg.datasets.split_utils import multimol_split, n_fold_multimol_split
from sklearn.model_selection import KFold


np.random.seed(15677)

# 100 randomly named molecules and frame numbers
mol_names = np.array(["{:04d}".format(i) for i in range(0, 100)])
mol_dict = {name: np.random.randint(10, high=50) for name in mol_names}

# test 3-fold cross validation sets over molecule names
folder = KFold(3, shuffle=True, random_state=42)
train_lists = []
val_lists = []
for train_idx, val_idx in folder.split(mol_names):
    train_lists.append(mol_names[train_idx])
    val_lists.append(mol_names[val_idx])


def test_named_split():
    """Tests splits that are explicitly named"""
    split_dict = multimol_split(
        mol_dict, train_names=train_lists[0], val_names=val_lists[0]
    )
    assert set(split_dict["train"]) == set(train_lists[0])
    assert set(split_dict["val"]) == set(val_lists[0])


def test_named_split_error_raise():
    """Tests to make sure ValueError is raised if named splits are not both None nor both np.ndarray"""
    with pytest.raises(ValueError):
        multimol_split(mol_dict, train_names=train_lists[0])
    with pytest.raises(ValueError):
        multimol_split(mol_dict, val_names=val_lists[0])


def test_named_split_warning():
    """Tests to make sure the user is warned if there is set leakage or non-comprehensive coverage"""
    # leaky sets
    with pytest.warns(UserWarning):
        multimol_split(
            mol_dict,
            train_names=train_lists[0],
            val_names=np.append(val_lists[0], train_lists[0][0]),
        )

    # coverage
    with pytest.warns(UserWarning):
        multimol_split(
            mol_dict,
            train_names=train_lists[0],
            val_names=np.delete(val_lists[0], 0),
        )


def test_mol_split():
    """Tests to make sure molecules are split properly"""
    split_dict = multimol_split(mol_dict)
    assert len(set(split_dict["train"]).intersection(split_dict["val"])) == 0


def test_kfold_split():
    """Tests to make sure non-overlapping K folds are made properly"""

    k_fold_splits = n_fold_multimol_split(
        mol_dict, k=3, shuffle=True, random_state=42
    )
    for i in range(3):
        fold_split = k_fold_splits["fold_{}".format(i)]
        assert (
            len(set(fold_split["train"]).intersection(fold_split["val"])) == 0
        )
