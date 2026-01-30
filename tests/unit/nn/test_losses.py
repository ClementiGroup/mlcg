from copy import deepcopy
from typing import Dict
import torch
import pytest
import numpy as np
from torch_geometric.data.collate import collate
from mlcg.data._keys import ENERGY_KEY, FORCE_KEY
from mlcg.data.atomic_data import AtomicData
from mlcg.nn.losses import Loss, ForceRMSE

simple_prediction = torch.tensor(
    [[1, 3, 1], [0, 0, 1], [6, 8, 2]],
)
simple_reference = torch.tensor(
    [[9, 5, 6], [7, 7, 0], [1, 3, 0]],
)

data = AtomicData.from_points(
    pos=torch.zeros((3, 3)),
    atom_types=torch.ones((3)),
    forces=simple_reference,
    masses=torch.ones((3)),
)

data.out = {}
data.out[FORCE_KEY] = simple_prediction


@pytest.mark.parametrize(
    "loss_func_call, loss_args, loss_kwargs, in_data, expected_loss",
    [
        (
            ForceRMSE,
            [],
            {},
            data,
            torch.sqrt(((data[FORCE_KEY] - data.out[FORCE_KEY]) ** 2).mean()),
        ),
        (
            ForceRMSE,
            [],
            {"reduction": "sum"},
            data,
            torch.sqrt(((data[FORCE_KEY] - data.out[FORCE_KEY]) ** 2).sum()),
        ),
        (
            Loss,
            [[ForceRMSE(), ForceRMSE()]],
            {},
            data,
            2
            * torch.sqrt(((data[FORCE_KEY] - data.out[FORCE_KEY]) ** 2).mean()),
        ),
        (
            Loss,
            [[ForceRMSE(), ForceRMSE()]],
            {"weights": [1.2, 0.5]},
            data,
            1.7
            * torch.sqrt(((data[FORCE_KEY] - data.out[FORCE_KEY]) ** 2).mean()),
        ),
    ],
)
def test_generalized_loss(
    loss_func_call, loss_args, loss_kwargs, in_data, expected_loss
):
    """Test to make sure that different losses with different options
    produce the correct outputs.
    """

    loss_function = loss_func_call(*loss_args, **loss_kwargs)
    loss = loss_function(in_data).numpy()
    np.testing.assert_array_equal(expected_loss.numpy(), loss)
