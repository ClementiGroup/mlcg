import torch
import pytest
from mlcg.nn.radial_basis import (
    GaussianBasis,
    ExpNormalBasis,
    RIGTOBasis,
    SpacedExpBasis,
)
from mlcg.nn.cutoff import IdentityCutoff, CosineCutoff, ShiftedCosineCutoff


@pytest.mark.parametrize(
    "basis_type", [GaussianBasis, ExpNormalBasis, RIGTOBasis]
)
def test_cutoff_error_raise(basis_type):
    """Test to make sure that RBFs enforce sensible cutoffs"""
    with pytest.raises(ValueError):
        basis_type(cutoff=IdentityCutoff(cutoff_lower=10, cutoff_upper=0))


@pytest.mark.parametrize(
    "basis_type, kwargs, error_type",
    [(SpacedExpBasis, {"cutoff": 5.0, "sigma_factor": 0.5}, ValueError)],
)
def test_generic_error_raise(basis_type, kwargs, error_type):
    """Test to make sure that RBFs enforce sensible cutoffs"""
    with pytest.raises(error_type):
        basis_type(**kwargs)


@pytest.mark.parametrize(
    "basis_type, default_cutoff",
    [
        (GaussianBasis, IdentityCutoff),
        (ExpNormalBasis, CosineCutoff),
        (RIGTOBasis, ShiftedCosineCutoff),
    ],
)
def test_cutoff_defaults(basis_type, default_cutoff):
    cutoff_upper = 10
    basis = basis_type(cutoff=cutoff_upper)
    assert isinstance(basis.cutoff, default_cutoff)
    assert basis.cutoff.cutoff_lower == 0
    assert basis.cutoff.cutoff_upper == cutoff_upper
