from typing import List, Callable, Dict
import tempfile
from ase.build import molecule
import torch
import numpy as np
import pytest
from ase.atoms import Atoms
from torch_geometric.data.collate import collate


from mlcg.data._keys import ENERGY_KEY, FORCE_KEY
from mlcg.data.atomic_data import AtomicData
from mlcg.geometry import Topology
from mlcg.geometry.topology import get_connectivity_matrix, get_n_paths
from mlcg.geometry.statistics import fit_baseline_models
from mlcg.neighbor_list.neighbor_list import make_neighbor_list
from mlcg.nn.prior import HarmonicBonds, HarmonicAngles, Dihedral
from mlcg.nn.gradients import SumOut, GradientsOut


@pytest.fixture
def get_initial_data():
    def data_list_builder(
        mol: Atoms,
        nls: Dict,
        corruptor: Callable = None,
        add_masses=True,
    ) -> List[AtomicData]:
        """Helper function to generate broken data lists

        Parameters
        ----------
        mol:
            ASE molecule
        nls:
            Neighbor list dictionary
        corruptor:
            Anonynous (lambda) function that takes the current
            frame of the data list and conditionally returns
            different values. If corruptor is None, the returned
            data list will be assembled correctly.
        add_masses:
            If True, masses are specified in each AtomicData instance
            according to the ASE molecule

        Returns
        -------
        initial_data_list:
            List of AtomicData instances that has been corrupted
            at the frame and with the damage specified by the
            the corruptor. If there is no corruptor, then the data
            list will be properly constructed.
        """

        input_masses = lambda x: torch.tensor(mol.get_masses()) if x else None

        initial_data_list = []
        for frame in range(5):
            data_point = AtomicData(
                pos=torch.tensor(mol.get_positions()),
                atom_types=torch.tensor(mol.get_atomic_numbers()),
                masses=input_masses(add_masses),
                cell=None,
                velocities=None,
                neighbor_list=nls,
            )
            initial_data_list.append(data_point)

        if corruptor != None:
            # corrupt a frame
            for frame in range(5):
                corrupted_data, corrupted_key = corruptor(frame, mol)
                initial_data_list[frame][corrupted_key] = corrupted_data
        return initial_data_list

    return data_list_builder


@pytest.fixture
def ASE_prior_model():
    def _model_builder(mol: str = "CH3CH2NH2") -> Dict:
        """Fixture that returns a simple prior-only model of
        an ASE molecule with HarmonicBonds and HarmonicAngles
        priors whose parameters are estimated from coordinates
        artifically perturbed by some small Gaussian noise.

        Parameters
        ----------
        mol:
            Molecule specifying string found in ase.build.molecule
        out_targets:
            Targets the the model should predict

        Returns
        -------
        model_with_data:
            Dictionary that contains the fitted prior-only model
            under the key "model" and the noisey data, as a collated
            AtomicData instance, used for fitting under the key
            "collated_prior_data"
        """

        # Seeding
        rng = np.random.default_rng(94834)

        # Physical units
        temperature = 350  # K
        #:Boltzmann constan in kcal/mol/K
        kB = 0.0019872041
        beta = 1 / (temperature * kB)

        # Here we make a simple prior-only model of aluminum-fluoride
        # mol = molecule("AlF3")
        # Implement molecule with dihedrals
        mol = molecule(mol)
        test_topo = Topology.from_ase(mol)

        # Add in molecule with dihedral and compute edges
        conn_mat = get_connectivity_matrix(test_topo)
        dihedral_paths = get_n_paths(conn_mat, n=4)
        test_topo.dihedrals_from_edge_index(dihedral_paths)

        n_atoms = len(test_topo.types)
        initial_coords = np.array(mol.get_positions())

        prior_data_frames = []
        for i in range(1000):
            perturbed_coords = initial_coords + 0.2 * rng.standard_normal(
                initial_coords.shape
            )
            prior_data_frames.append(torch.tensor(perturbed_coords))
        prior_data_frames = torch.stack(prior_data_frames, dim=0)

        # Set up some data with bond/angle neighborlists:
        bond_edges = test_topo.bonds2torch()
        angle_edges = test_topo.angles2torch()
        dihedral_edges = test_topo.dihedrals2torch()

        # Generete some noisy data for the priors
        nls_tags = ["bonds", "angles", "dihedrals"]
        nls_orders = [2, 3, 4]
        nls_edges = [bond_edges, angle_edges, dihedral_edges]

        neighbor_lists = {
            tag: make_neighbor_list(tag, order, edge_list)
            for (tag, order, edge_list) in zip(nls_tags, nls_orders, nls_edges)
        }

        prior_data_list = []
        for frame in range(prior_data_frames.shape[0]):
            data_point = AtomicData(
                pos=prior_data_frames[frame],
                atom_types=torch.tensor(test_topo.types),
                masses=torch.tensor(mol.get_masses()),
                cell=None,
                neighbor_list=neighbor_lists,
            )
            prior_data_list.append(data_point)

        collated_prior_data, _, _ = collate(
            prior_data_list[0].__class__,
            data_list=prior_data_list,
            increment=True,
            add_batch=True,
        )
        # Fit the priors
        prior_cls = [HarmonicBonds, HarmonicAngles, Dihedral]
        priors, stats = fit_baseline_models(
            collated_prior_data, beta, prior_cls
        )

        # Construct the model
        priors = {
            name: GradientsOut(priors[name], targets=[FORCE_KEY])
            for name in priors.keys()
        }

        full_model = SumOut(priors)
        model_with_data = {
            "model": full_model,
            "collated_prior_data": collated_prior_data,
            "molecule": mol,
            "num_examples": len(prior_data_list),
            "neighbor_lists": neighbor_lists,
        }
        return model_with_data

    return _model_builder
