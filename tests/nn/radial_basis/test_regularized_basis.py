import pytest
import torch
from mlcg.nn.radial_basis import (
    GaussianBasis,
    ExpNormalBasis,
    RIGTOBasis,
    SpacedExpBasis,
    RegularizedBasis,
)
from mlcg.nn import (
    IdentityCutoff,
)


@pytest.mark.parametrize(
    "basis_type", [GaussianBasis, ExpNormalBasis, RIGTOBasis, SpacedExpBasis]
)
def test_cutoff_error_raise(basis_type):
    """Test to make sure that RBFs enforce sensible cutoffs"""
    with pytest.raises(ValueError):
        RegularizedBasis(
            basis_function=basis_type(
                cutoff=IdentityCutoff(cutoff_lower=10, cutoff_upper=0)
            ),
            types=5,
            n_basis_set=3,
        )


@pytest.mark.parametrize(
    "basis_type, num_rbf, n_basis_set, independent_regularizations",
    [
        (GaussianBasis, 10, 2, False),
        (GaussianBasis, 10, 1, False),
        (ExpNormalBasis, 8, 2, False),
        (ExpNormalBasis, 8, 1, False),
        (GaussianBasis, 10, 2, True),
        (GaussianBasis, 10, 1, True),
        (ExpNormalBasis, 8, 2, True),
        (ExpNormalBasis, 8, 1, True),
    ],
)
def test_regularized_basis_forward(
    basis_type, num_rbf, n_basis_set, independent_regularizations
):
    basis = RegularizedBasis(
        basis_function=basis_type(cutoff=5.0, num_rbf=num_rbf),
        types=4,
        n_basis_set=n_basis_set,
        independent_regularizations=independent_regularizations,
    )
    distances = torch.tensor([0.5, 1.0, 2.0, 2.0])
    type_i = torch.tensor([0, 1, 2, 3])
    type_j = torch.tensor([1, 2, 3, 2])
    output = basis(distances, type_i, type_j)
    assert output.shape == (n_basis_set, int(distances.shape[0]), num_rbf)
    assert not torch.any(torch.isnan(output))
    assert not torch.any(torch.isinf(output))
    reg_params = basis.get_regularization_parameters()
    if not independent_regularizations:
        assert reg_params.shape == (4 * (4 + 1) // 2, num_rbf)
    else:
        assert reg_params.shape == (n_basis_set, 4 * (4 + 1) // 2, num_rbf)


@pytest.mark.parametrize(
    "basis_type, n_basis_set, independent_regularizations, init_val",
    [
        (GaussianBasis, 2, False, 1.0),
        (GaussianBasis, 1, False, 1.0),
        (GaussianBasis, 2, True, 1.0),
        (GaussianBasis, 1, True, 1.0),
        (GaussianBasis, 2, True, [1.0, 2.0]),
        (GaussianBasis, 1, True, [1.0, 2.0]),
    ],
)
def test_regularized_basis_symmetry(
    basis_type, n_basis_set, independent_regularizations, init_val
):
    basis = RegularizedBasis(
        basis_function=basis_type(cutoff=5.0, num_rbf=10),
        types=4,
        n_basis_set=n_basis_set,
        independent_regularizations=independent_regularizations,
        init_val=init_val,
    )
    distances = torch.tensor([0.5, 1.0, 2.0, 2.0])
    type_i = torch.tensor([0, 1, 2, 3])
    type_j = torch.tensor([1, 2, 3, 2])
    output = basis(distances, type_i, type_j)

    # test that regularization acts in the same way for inverted indeces
    assert torch.allclose(
        output[:, 2, :],
        output[:, 3, :],
        atol=1e-6,
    )
