import torch
import pytest
import numpy as np

from mlcg.nn.cutoff import IdentityCutoff, CosineCutoff, ShiftedCosineCutoff

data = torch.randn(100, 3)
linear_data = torch.linspace(0, 10, 100)


@pytest.mark.parametrize("cutoff_type", [IdentityCutoff, CosineCutoff])
def test_cutoff_error_raise(cutoff_type):
    """Test to make sure that cutoffs enforce sensible cutoffs"""
    with pytest.raises(ValueError):
        cutoff_type(cutoff_lower=10, cutoff_upper=0)


def test_identity_cutoff():
    """Test to make sure that IdentityCutoff performs an identity transform"""
    cutoff = IdentityCutoff()
    data_out = cutoff(data)
    np.testing.assert_array_equal(
        torch.ones_like(data).numpy(), data_out.numpy()
    )


@pytest.mark.parametrize(
    "cutoff_lower, cutoff_upper, expected_lower_value, expected_upper_value",
    [(0, 5, 1, 0), (5, 10, 0, 0)],
)
def test_cosine_cutoff(
    cutoff_lower, cutoff_upper, expected_lower_value, expected_upper_value
):
    """Test to make sure that CosineCutoff produces the correct cutoff values at the endpoints
    for lower cutoffs of zero and greater than zero
    """
    cutoff = CosineCutoff(cutoff_lower=cutoff_lower, cutoff_upper=cutoff_upper)
    cutoff_data = cutoff(linear_data).numpy()
    print(cutoff_data[0])
    print(cutoff_data[1])
    assert cutoff_data[0] == expected_lower_value
    assert cutoff_data[-1] == expected_upper_value


@pytest.mark.parametrize(
    "cutoff_upper, expected_upper_value",
    [(5, 0), (10, 0)],
)
def test_shifted_cosine_cutoff(cutoff_upper, expected_upper_value):
    """Test to make sure that CosineCutoff produces the correct cutoff values at the endpoints
    for lower cutoffs of zero and greater than zero
    """
    cutoff = ShiftedCosineCutoff(cutoff=cutoff_upper)
    cutoff_data = cutoff(linear_data).numpy()
    print(cutoff_data)
    assert cutoff_data[0] == 1
    assert cutoff_data[-1] == expected_upper_value
