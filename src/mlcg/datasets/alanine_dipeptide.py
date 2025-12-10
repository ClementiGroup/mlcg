import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.collate import collate
import mdtraj
import pickle
from typing import List, Tuple

from ..utils import tqdm, download_url
from ..geometry.topology import Topology
from ..geometry.statistics import fit_baseline_models
from ..cg import build_cg_matrix, build_cg_topology, AL_CG_MAP
from ..data import AtomicData
from ..nn import (
    HarmonicBonds,
    HarmonicAngles,
    Repulsion,
    GradientsOut,
)
from ..nn.prior import _Prior
from .utils import remove_baseline_forces, chunker


class AlanineDataset(InMemoryDataset):
    r"""Dataset for training a CG model of the alanine-dipeptide protein following a Cα + 1 C\beta CG mapping

    Alanine Dipeptide CG structure:

    .. code-block::

                CB(3)
                  |
          N(1) - CA(2) - C(4)
         /                  \
        C(0)                 N(5)


    This Dataset produces delta forces for model training, in which the CG prior forces (harmonic bonds and angles) have been subtracted from the full CG forces.

    The class works as follows:
        - If the raw data (coordinates, forces, and pdb file) for alanine dipeptide does not exist, the files will automatically be downloaded and processed
        - If the raw data exists, "root/processed/" will be created and the raw dataset will be processed
        - If the raw data and processed data exists, the class will load the processed dataset containing:
            - data : AtomicData object containing all the CG positions, forces, embeddings
            - slices : Slices of the AtomicData object
            - prior_models : Prior models (HarmonicBonds, HarmonicAngles) of the dataset with x_0 and \sigma
            - topologies : Object of Topology class containing all topology information of the _CG_ molecule, including neighbor lists for bond and angle priors

    Inputs:
        root :
            Location for AlanineDataset to be downloaded and processed
        stride :
            Stride length over dataset

    Default priors:
        - HarmonicBonds, HarmonicAngles
    Optional priors:
        - Repulsion
            - If repulsion prior is used, a custom neighbor list is created for the repulsion prior where all pairs of beads
              not interacting through bonds and angles are included in the interaction set
    """

    #:Temperature used to generate the underlying all-atom data in [K]
    temperature = 300  # K
    #:Boltzmann constant in kcal/mol/K
    kB = 0.0019872041
    #:
    _priors_cls = [HarmonicBonds, HarmonicAngles]

    def __init__(
        self,
        root: str,
        stride: int = 1,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.stride = stride
        self.priors_cls = AlanineDataset._priors_cls
        self.beta = 1 / (AlanineDataset.temperature * AlanineDataset.kB)

        super(AlanineDataset, self).__init__(
            root, transform, pre_transform, pre_filter
        )

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.topologies = {"ala2": torch.load(self.processed_paths[1])}
        self.prior_models = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        return [
            "alanine-dipeptide-1Mx1ps-with-force.npz",
            "al2.pdb",
        ]

    @property
    def processed_file_names(self):
        return ["ala2.pt", "topologies.pt", "priors.pt"]  # , "al2.pdb"]

    def download(self):
        # Download to `self.raw_dir`.
        url_trajectory = "http://ftp.mi.fu-berlin.de/pub/cmb-data/alanine-dipeptide-1Mx1ps-with-force.npz"
        url_pdb = "http://ftp.mi.fu-berlin.de/pub/cmb-data/al2.pdb"
        path_inputs = download_url(url_trajectory, self.raw_dir)
        path_pdb = download_url(url_pdb, self.raw_dir)

    def make_data_slices_prior(
        self,
        coords_forces_file: str,
        topology: Topology,
        cg_mapping: dict,
        cg_topo: Topology,
    ) -> Tuple[AtomicData, dict, torch.nn.ModuleDict]:
        """Method to make collated AtomicData object, slices, and baseline models

        Parameters
        ----------
        coords_forces_file:
            npz file containing the forces and coordinates from the all-atom simulation
        topology:
            Topology of all-atom model
        cg_mapping:
            Dictionary containing CG mapping.
        cg_topo:
            Topology of CG model

        Returns
        -------
        data_list_coll:
            Collated AtomicData object
        slices:
            Dictionary containing slices of the AtomicData object
        baseline_models:
            Module dictionary containing fitted prior models

        """
        ## LOAD DATA FILE AND ALLOCATE COORDINATES AND FORCES
        data = np.load(coords_forces_file)
        coords = data["coordinates"]
        forces = data["forces"]

        ## MAKE EMBEDDINGS AND MASSES AND CG_MATRIX
        embeddings, masses, cg_matrix, cg_mapping_ = build_cg_matrix(
            topology, cg_mapping=cg_mapping, special_terminal=False
        )

        n_beads = cg_matrix.shape[0]

        ## MAKE PRIOR NEIGHBOR LIST
        prior_nls = self.make_priors(self.priors_cls, cg_topo)

        ## MAKE ATOMICDATA OBJECT
        data_list = []

        # MAKE CG COORDINATES AND CG FORCES
        cg_coords = np.array(
            np.einsum("mn, ind-> imd", cg_matrix, coords), dtype=np.float32
        )

        f_proj = np.dot(
            np.linalg.inv(np.dot(cg_matrix, cg_matrix.T)), cg_matrix
        )

        cg_forces = np.array(
            np.einsum("mn, ind-> imd", f_proj, forces), dtype=np.float32
        )

        for i, (cg_coord, cg_force) in enumerate(
            zip(cg_coords[:: self.stride], cg_forces[:: self.stride])
        ):
            pos = torch.from_numpy(cg_coord)
            z = torch.from_numpy(embeddings).long()
            force = torch.from_numpy(cg_force)

            ## MAKE ATOMICDATA OBJECT
            atomicData = AtomicData.from_points(
                atom_types=z,
                pos=pos,
                forces=force,
                masses=masses,
                neighborlist=prior_nls,
                traj_id=0,
                name="ala2",
                frame_id=i,
            )

            data_list.append(atomicData)

        ## COLLATE ATOMICDATA DATASET
        data_list_coll, _, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=True,
            add_batch=True,
        )

        ## MAKE BASELINE MODELS
        baseline_models = self.make_baseline_models(
            data_list_coll, self.beta, self.priors_cls
        )

        ## REMOVE BASELINE FORCES
        ## Important -- This removes baseline forces in place rather than explicitly by returning values
        ## from the method
        batch_size = 512
        chunks = tuple(chunker(data_list, batch_size))
        for sub_data_list in tqdm(chunks, "Removing baseline forces"):
            _ = remove_baseline_forces(
                sub_data_list,
                baseline_models,
            )

        ## COLLATE ATOMIC DATASET WITHOUT INCREMENT
        data_list_coll, slices, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=False,
            add_batch=False,
        )

        ## REMOVE NEIGHBOR LIST AND BASELINE FORCES
        delattr(data_list_coll, "baseline_forces")
        data_list_coll.neighbor_list = {}

        return data_list_coll, slices, baseline_models

    def make_priors(
        self,
        priors_cls: List[_Prior],
        cg_topo: Topology,
    ) -> dict:
        """Method to make prior neighbor lists


        Parameters
        ----------
        priors_cls:
            List of prior classes
        cg_topo:

        Returns
        -------
        prior_nls:
            Dictionary containing all prior neighbor lists

        """
        prior_nls = {}
        for cl in priors_cls:
            if cl is not Repulsion:
                prior_nls.update(**cl.neighbor_list(cg_topo))
            elif cl is Repulsion:
                prior_nls.update(self.repulsion_nls(cl.name, cg_topo))
        return prior_nls

    @staticmethod
    def repulsion_nls(name: str, cg_topo: Topology) -> dict:
        """Method for generating neighbor list for Repulsion prior - all interactions not included in bond or angle interactions


        Parameters
        ----------
        name:
            Name of prior class
        cg_topo:
            Topology object of CG molecule

        Returns
        -------
        dict
            Dictionary containing repulsion neighor list

        """
        ids = torch.arange(cg_topo.n_atoms)

        ## MAKE MAPPING OF ALL PAIRS OF ATOMS
        mapping = torch.cartesian_prod(ids, ids).t()
        ## REMOVE MAPPING WITH SELF
        mapping = mapping[:, mapping[0] != mapping[1]]

        ## REMOVE BONDED MAPPINGS FROM REPULSION MAPPING
        for bond_num in range(len(cg_topo.bonds[0])):
            atom_0 = cg_topo.bonds[0][bond_num]
            atom_1 = cg_topo.bonds[1][bond_num]
            mask_0 = (mapping[0, :] == atom_0) + (mapping[1, :] == atom_0)
            mask_1 = (mapping[0, :] == atom_1) + (mapping[1, :] == atom_1)

            mapping = mapping[:, ~(mask_0 * mask_1)]

        ## REMOVE ANGLE MAPPINGS FROM REPULSION MAPPING
        for angle_num in range(len(cg_topo.angles[0])):
            atom_0 = cg_topo.angles[0][angle_num]
            atom_1 = cg_topo.angles[2][angle_num]
            mask_0 = (mapping[0, :] == atom_0) + (mapping[1, :] == atom_0)
            mask_1 = (mapping[0, :] == atom_1) + (mapping[1, :] == atom_1)

            mapping = mapping[:, ~(mask_0 * mask_1)]

        ## MAKE NEIGHBOR LIST IN STANDARD FORMAT USING MAKE_NEIGHBOR_LIST
        from mlcg.neighbor_list.neighbor_list import make_neighbor_list

        nl = make_neighbor_list(
            tag="fully connected",
            order=mapping.shape[0],
            index_mapping=mapping,
            self_interaction=False,
        )

        return {name: nl}

    @staticmethod
    def load_original_topology(pdb_file: str) -> Topology:
        """Method to load origin topology


        Parameters
        ----------
        pdb_file:
            Path of all-atom PDB file

        Returns
        -------
        topology:
            Topology class object for all-atom molecule

        """
        # LOAD TRAJECTORY AND REMOVE SOLVENT
        topo = mdtraj.load(pdb_file).remove_solvent().topology

        ## MAKE TOPOLOGY OBJECT FROM MDTRAJ FILE
        topology = Topology.from_mdtraj(topo)

        return topology

    @staticmethod
    def make_cg_topology(
        topology: Topology,
        cg_mapping: dict = AL_CG_MAP,
        special_terminal: bool = False,
    ) -> Topology:
        """Method to make Topology class object of CG molecule, creates custom bonds and angles to make a non-linear CG molecule

        Parameters
        ----------
        topology:
            All-atom topology
        cg_mapping : (optional)
            Dictionary containing CG mapping. The default is AL_CG_MAP.
        special_terminal : (optional)
            True if termini beads are to be treated separately. The default is False.

        Returns
        -------
        cg_topo:
            Topology class object for CG molecule

        """
        ## BUILD CG TOPOLOGY USING CG MAPPING
        cg_topo = build_cg_topology(
            topology, cg_mapping=cg_mapping, special_terminal=special_terminal
        )

        ## MODIFY CG TOPOLOGY TO MAKE CORRECT TOPOLOGY
        ## Function build_cg_topology makes a linear chain topology by default, hence some modifications are required to make a non-linear chain molecule
        ## Hard-coded values for
        ## 1. removing 3-4 bond (where 3 is beta-carbon)
        ## 2. adding 2-4 bond
        ## 3. removing angles [2,3,4] and [3,4,5]
        ## 4. adding angles [2,4,5] (along backbone), [3,2,4] (carbon-beta angle), [1,2,4] (along backbone)

        cg_topo.remove_bond([[3, 4]])  # 1
        cg_topo.add_bond(2, 4)  # 2

        cg_topo.remove_angle([[2, 3, 4], [3, 4, 5]])  # 3

        cg_topo.add_angle(2, 4, 5)  # 4
        cg_topo.add_angle(3, 2, 4)  # 4
        cg_topo.add_angle(1, 2, 4)  # 4

        return cg_topo

    @staticmethod
    def make_baseline_models(
        data: AtomicData,
        beta: float,
        priors_cls: List[_Prior],
    ) -> torch.nn.ModuleDict:
        """Method to make all baseline models


        Parameters
        ----------
        data:
            AtomicData object of entire trajectory
        beta:
            1/(k_B * T)
        priors_cls:
            List of prior classes

        Returns
        -------
        baseline_models:
            Dictionary of prior models fitted with the right harmonic restraint values

        """
        baseline_models, statistics = fit_baseline_models(
            data, beta, priors_cls
        )

        ## TAKE GRADIENTS THROUGH BASELINE MODELS
        for k in baseline_models.keys():
            baseline_models[k] = GradientsOut(
                baseline_models[k], targets="forces"
            )

        return baseline_models

    def save_dataset(self, pickle_name: str):
        """Method for saving dataset given pickle name

        Parameters
        ----------
        pickle_name:
            Name of pickle to store the dataset in

        """
        with open(pickle_name, "wb") as f:
            pickle.dump(self, f)

    def process(
        self,
        cg_mapping: dict = AL_CG_MAP,
    ):
        """Method for processing the raw data - this is where all processing function calls take place
        All outputs are stored in the relevant processed files


        Parameters
        ----------
        cg_mapping: (optional)
            CG mapping dictionary. The default is AL_CG_MAP.


        """
        ## LOAD TOPOLOGY USING PDB FILE
        topology = self.load_original_topology(self.raw_paths[1])

        ## MAKE CG TOPOLOGY
        topologies = self.make_cg_topology(
            topology, cg_mapping=cg_mapping, special_terminal=False
        )

        ## MAKE ATOMIC DATA OBJECT AND ATOM SLICES
        data, slices, prior_models = self.make_data_slices_prior(
            self.raw_paths[0],
            topology,
            cg_mapping,
            topologies,
        )

        ## SAVE PROCESSED OUTPUT
        torch.save((data, slices), self.processed_paths[0])
        torch.save((topologies), self.processed_paths[1])
        torch.save(prior_models, self.processed_paths[2])
