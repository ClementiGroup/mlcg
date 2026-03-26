import pytest
import torch

from mlcg.nn.prior.repulsion import Repulsion, FlashRepulsion

from mlcg.geometry import compute_statistics, fit_baseline_models
from mlcg.nn import GradientsOut

from mlcg.mol_utils import MolDatabase
from mlcg.neighbor_list.neighbor_list import atomic_data2neighbor_list

CUTOFF_UPPER = 4.0
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def create_data(prior_name):
    database = MolDatabase()
    data = database.collated_data

    if prior_name == Repulsion.name:
        data.neighbor_list = {
            prior_name: atomic_data2neighbor_list(
                database.collated_data,
                CUTOFF_UPPER,
                self_interaction=False,
                max_num_neighbors=1000,
                nls_distance_method="torch",
            )
        }

    else:
        raise NotImplementedError(
            f"Test doesn't support prior {prior_name}, please add neighborlist support in create_data function"
        )

    return data


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "Base_prior, Flash_prior", [(Repulsion, FlashRepulsion)]
)
def test_repulsion_kernel(device, Base_prior, Flash_prior):

    data = create_data(Base_prior.name)

    stat = compute_statistics(
        data,
        Base_prior.name,
        1.6,
        Base_prior,
        bmin=0,
        bmax=10,
        fit_from_values=True,
    )

    base_prior = GradientsOut(Base_prior(stat))
    flash_prior = GradientsOut(
        Flash_prior(base_prior.model.sigma, Base_prior.name)
    )
    compiled_flash_prior = torch.compile(flash_prior)

    data = data.to(device)
    base_prior = base_prior.to(device)
    flash_prior = flash_prior.to(device)

    data = base_prior(data)
    base_energy = data.out[Repulsion.name]["energy"]
    base_forces = data.out[Repulsion.name]["forces"]

    data.out = {}
    data = flash_prior(data)
    flash_energy = data.out[Repulsion.name]["energy"]
    flash_forces = data.out[Repulsion.name]["forces"]

    torch.testing.assert_close(base_energy, flash_energy)
    torch.testing.assert_close(base_forces, flash_forces)

    if device == "cuda":
        data.out = {}
        data = compiled_flash_prior(data)
        compiled_flash_energy = data.out[Repulsion.name]["energy"]
        compiled_flash_forces = data.out[Repulsion.name]["forces"]

        torch.testing.assert_close(base_energy, compiled_flash_energy)
        torch.testing.assert_close(base_forces, compiled_flash_forces)
