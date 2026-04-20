import pytest
import torch

from mlcg.nn.prior import Repulsion, HarmonicBonds, HarmonicAngles, Dihedral
from mlcg.geometry import compute_statistics
from mlcg.nn import GradientsOut

from mlcg.mol_utils import MolDatabase, _ASE_prior_model
from mlcg.neighbor_list.neighbor_list import atomic_data2neighbor_list
from mlcg.nn.kernels.converter import convert_standard_model_to_flash

CUTOFF_UPPER = 4.0
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def create_data(prior):
    if prior.name == Repulsion.name:
        database = MolDatabase()
        data = database.collated_data
        data.neighbor_list = {
            prior.name: atomic_data2neighbor_list(
                database.collated_data,
                CUTOFF_UPPER,
                self_interaction=False,
                max_num_neighbors=1000,
                nls_distance_method="torch",
            )
        }
        stat = compute_statistics(
            data,
            prior.name,
            1.6,
            prior,
            bmin=0,
            bmax=10,
            fit_from_values=True,
        )

        base_prior = GradientsOut(prior(stat))

    elif prior.name in [HarmonicBonds.name, HarmonicAngles.name, Dihedral.name]:
        model_with_data = _ASE_prior_model(sum_out=False)
        data = model_with_data["collated_prior_data"]
        base_prior = model_with_data["model"][prior.name]

    else:
        raise NotImplementedError(
            f"Test doesn't support prior {prior.name}, please add neighborlist support in create_data function"
        )

    flash_prior = convert_standard_model_to_flash(base_prior)
    return data, base_prior, flash_prior


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "base_prior",
    [
        (Repulsion),
        (HarmonicBonds),
        (HarmonicAngles),
        (Dihedral),
    ],
)
def test_base_vs_kernel(device, base_prior):

    data, base_prior, flash_prior = create_data(base_prior)

    compiled_flash_prior = torch.compile(flash_prior)

    data = data.to(device)
    base_prior = base_prior.eval().to(device)
    flash_prior = flash_prior.eval().to(device)
    compiled_flash_prior = compiled_flash_prior.eval().to(device)

    data = base_prior(data)
    base_energy = data.out[base_prior.name]["energy"]
    base_forces = data.out[base_prior.name]["forces"]

    data.out = {}
    data = flash_prior(data)
    flash_energy = data.out[base_prior.name]["energy"]
    flash_forces = data.out[base_prior.name]["forces"]
    torch.testing.assert_close(base_energy, flash_energy)
    torch.testing.assert_close(base_forces, flash_forces)

    if device == "cuda":
        data.out = {}
        data = compiled_flash_prior(data)
        compiled_flash_energy = data.out[base_prior.name]["energy"]
        compiled_flash_forces = data.out[base_prior.name]["forces"]

        torch.testing.assert_close(base_energy, compiled_flash_energy)
        torch.testing.assert_close(base_forces, compiled_flash_forces)
