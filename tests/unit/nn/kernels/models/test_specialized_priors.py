import pytest
import torch

from torch_geometric.data.collate import collate

from mlcg.nn.prior import (
    HarmonicBonds,
    HarmonicAngles,
    Dihedral,
)
from mlcg.nn import SumOut
from mlcg.mol_utils import _ASE_prior_model
from mlcg.simulation.specialize_prior import condense_all_priors_for_simulation
from mlcg.nn.kernels.converter import convert_standard_model_to_flash

CUTOFF_UPPER = 4.0
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def create_data(prior):
    if prior.name not in [
        HarmonicBonds.name,
        HarmonicAngles.name,
        Dihedral.name,
    ]:
        raise NotImplementedError(
            f"Test doesn't support prior {prior.name}, please add neighborlist support in create_data function"
        )

    model_with_data = _ASE_prior_model(sum_out=False)
    base_prior = SumOut(
        torch.nn.ModuleDict({prior.name: model_with_data["model"][prior.name]})
    )

    specialized_prior, data_list = condense_all_priors_for_simulation(
        base_prior, model_with_data["data_list"]
    )
    flash_specialized_prior = convert_standard_model_to_flash(specialized_prior)

    collated_specialized_data, _, _ = collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=True,
        add_batch=True,
    )

    return (
        model_with_data["collated_prior_data"],
        collated_specialized_data,
        base_prior,
        specialized_prior,
        flash_specialized_prior,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "base_prior",
    [
        (HarmonicBonds),
        (HarmonicAngles),
        (Dihedral),
    ],
)
def test_base_vs_kernel(device, base_prior):

    prior_name = base_prior.name

    (
        data,
        specialized_data,
        base_prior,
        specialized_prior,
        flash_specialized_prior,
    ) = create_data(base_prior)

    compiled_flash_specialized_prior = torch.compile(flash_specialized_prior)

    data = data.to(device)
    specialized_data = specialized_data.to(device)
    base_prior = base_prior.to(device)
    specialized_prior = specialized_prior.to(device)
    flash_specialized_prior = flash_specialized_prior.to(device)
    compiled_flash_specialized_prior = compiled_flash_specialized_prior.to(
        device
    )

    data = base_prior(data)
    base_energy = data.out[prior_name]["energy"]
    base_forces = data.out[prior_name]["forces"]

    specialized_data = flash_specialized_prior(specialized_data)
    specialized_energy = specialized_data.out[prior_name]["energy"]
    specialized_forces = specialized_data.out[prior_name]["forces"]
    torch.testing.assert_close(base_energy, specialized_energy)
    torch.testing.assert_close(base_forces, specialized_forces)

    specialized_data.out = {}
    specialized_data = flash_specialized_prior(specialized_data)
    flash_energy = specialized_data.out[prior_name]["energy"]
    flash_forces = specialized_data.out[prior_name]["forces"]
    torch.testing.assert_close(base_energy, flash_energy)
    torch.testing.assert_close(base_forces, flash_forces)

    if device == "cuda":
        specialized_data.out = {}
        specialized_data = compiled_flash_specialized_prior(specialized_data)
        compiled_flash_energy = specialized_data.out[prior_name]["energy"]
        compiled_flash_forces = specialized_data.out[prior_name]["forces"]

        torch.testing.assert_close(base_energy, compiled_flash_energy)
        torch.testing.assert_close(base_forces, compiled_flash_forces)
