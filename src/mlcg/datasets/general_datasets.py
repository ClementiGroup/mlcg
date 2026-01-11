from typing import Union, Optional, Callable, List, Tuple, Dict
import pickle
import warnings
import os
from os.path import join
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.collate import collate
import mdtraj as md
from ..utils import tqdm, download_url
from ..geometry.topology import (
    Topology,
    add_chain_bonds,
    add_chain_angles,
    add_chain_dihedrals,
)
from ..geometry.statistics import fit_baseline_models, compute_statistics
from ..data import AtomicData
from ..nn import (
    HarmonicBonds,
    HarmonicAngles,
    Repulsion,
    Dihedral,
    GradientsOut,
)
import pickle
from ..neighbor_list.neighbor_list import make_neighbor_list
from ..nn.prior import _Prior
from .utils import remove_baseline_forces, chunker
from itertools import combinations
import h5py


class GeneralCarbonAlphaDataset(InMemoryDataset):
    """
    General single-protein carbon alpha dataset built from torch_geometric's
    `InMemoryDataset`. Produces a CG H5 dataset and a set of fitted priors for
    a single chain, single molecule carbon alpha CG model for force matching.

    Please note the following:

    1. A change in priors or prior neighborlists requires a COMPLETE regeration of
       The H5 dataset. A specific prior and corresponding prior neighborlist can only
       be used for a single H5 dataset. CG simulations of any model trained on this H5
       dataset MUST use the same priors used to create said H5 dataset

    Parameters
    ----------
    root:
        directory that specifies where the processed dataset and priors should be saved
    coordmap:
        np.array of shape (num_cg_atoms, num_aa_atoms) representing a global CG coordinate
        mapping from the all-atom resolution to the CG resolution
    forcemap:
        np.array of shape (num_cg_atoms, num_aa_atoms) representing a global CG force
        mapping from the all-atom resolution to the CG resolution
    raw_data_files:
        List of tuples of all-atom dataset files. Each tuple should contain the string/path inputs
        that must be used by coord_force_loader
    atom_types:
        `np.array` of shape (num_cg_atoms,) specifying the integer type of each CG atom in sequence
    masses:
        `np.array` of shape (num_cg_atoms,) specifying the mass of each CG atom in sequence
    pdb_file:
        Path to all-atom PDB file, whose protein atom ordering corresponds to that found in the
        all-atom coordinate and force files
    coord_force_loader:
        Function that takes in arguments from the tuples in the `raw_data_files` list. Should return a
        tuple of `np.array`s containing the all-atom coordinates, and forces
    mol_name:
        String specifying the name of the molecule that you are modeling. Used for saving H5 datasets.
    precision:
        String that sets the precision level for CG coordinates and forces
    priors:
        If a `str` is specified, an existing prior model (torch.torch.nn.Module) will be loaded
        from that path. Else, a list of `_Prior` classes can be supplied for parametrization
        from reference data.
    prior_nls_file:
        If a prior file is specified for the `priors` arg, this file of pickled dictionary of
        `torch` neighborlists will be loaded for use.
    num_prior_samples:
        If a list of `_Prior` classes is specifed for the `priors` kwarg, this integer determines
        how many frames of the rerefence data should be randomly sampled (uniformly) for prior
        parametrization.
    num_sim_starts:
        Number of random reference frames for which CG configurations are saved as simulation starts
        for later simulation of trained models
    temperature:
        Temperature of the all-atom system in Kelvins. A Boltzmann constant in units of kcal/mol/K
        is used to produce a corresponding inverse thermodynamic temperature (i.e., "beta")
    non_bond_cut:
        If a `Repulsion` prior is specfied by the `priors` kwarg for parametrization, this integer
        determines how many nearest neighbors to exclude in the construction of the set of non-bonded
        pairs. For example, `non_bond_cut=2` means that CG atoms up to 2 nearest neighbors away
        (eg, bonds, and angles) are excluded from the non-bonded set.
    dihedral_fit_kwargs:
        If specified, all dihedral priors will be fit specifically using the specified method and degree.
        If `None`, default dihedral fitting procedures are adopted. See `help(mlcg.nn.prior.Dihedral)`
        and `help(mlcg.geometry.statistics.compute_statistics)`.
    exclude_cis_omega:
        If `True`, any reference trajectory with a cis omega backbone angle present will be removed
        from the dataset entirely.
    delta_force_batch_size:
        Number of frames in each batch for delta force production. For larger proteins, you may
        want to reduce this.
    delta_check_threshold:
        Detection for threshold of absolute value delta forces that might be too large from poor prior choices
    transform:
        see `help(torch_geometric.data.Dataset)`
    pre_transform:
        see `help(torch_geometric.data.Dataset)`
    pre_filter:
        see `help(torch_geometric.data.Dataset)`
    verbose:
        If `True`, tqdm bars and other diagnostic information will be displayed to STDOUT
    """

    #:Boltzmann constant in kcal/mol/K
    kB = 0.0019872041

    def __init__(
        self,
        root: str,
        coordmap: np.array,
        forcemap: np.array,
        raw_data_fns: List[Tuple[str]],
        atom_types: np.array,
        masses: np.array,
        pdb_file: str,
        coord_force_loader: Callable,
        mol_name: str = "MyMolecule",
        precision: str = "float32",
        priors: Union[str, List[_Prior]] = [
            HarmonicBonds,
            HarmonicAngles,
            Dihedral,
            Repulsion,
        ],
        prior_nls_file: Optional[str] = None,
        num_prior_samples: int = 1000000,
        num_sim_starts: int = 100,
        temperature: float = 350,
        non_bond_cut: int = 2,
        dihedral_fit_kwargs: Optional[Dict] = {"constrain_deg": 5, "n_degs": 5},
        exclude_cis_omega: bool = True,
        delta_force_batch_size: int = 256,
        delta_check_threshold: float = 100000.0,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        verbose: bool = True,
    ):
        self.temperature = temperature
        self.beta = 1 / (self.temperature * self.kB)
        self.coordmap = coordmap
        self.forcemap = forcemap
        self.num_cg_atoms = self.coordmap.shape[0]
        self.num_aa_atoms = self.coordmap.shape[1]
        self.atom_types = atom_types
        self.masses = masses
        self.raw_data_fns = raw_data_fns
        self.pdb_file = pdb_file
        self.coord_force_loader = coord_force_loader
        self.non_bond_cut = non_bond_cut
        self.num_prior_samples = num_prior_samples
        self.num_sim_starts = num_sim_starts
        self.exclude_cis_omega = exclude_cis_omega
        self.delta_force_batch_size = delta_force_batch_size
        self.delta_check_threshold = delta_check_threshold
        self.verbose = verbose
        self.mol_name = mol_name
        self.precision = precision

        assert self.temperature > 0
        assert self.non_bond_cut > 0
        assert self.atom_types.shape == self.masses.shape
        assert self.coordmap.shape == self.forcemap.shape
        assert len(self.coordmap.shape) == len(self.forcemap.shape) == 2
        assert isinstance(self.coord_force_loader, Callable)
        for filename_set in self.raw_data_fns:
            assert len(filename_set) == len(self.raw_data_fns[0])

        if isinstance(priors, list):
            assert all([issubclass(p, _Prior) for p in priors])
            self.prior_classes = priors
            self.dihedral_fit_kwargs = dihedral_fit_kwargs
            self.prior_file = None
            self.prior_nls_file = None
        elif isinstance(priors, str):
            if prior_nls_file == None:
                raise RuntimeError(
                    "Pre-existing priors specified, but no `prior_nls_file` specified"
                )
            self.prior_file = priors
            self.prior_nls_file = prior_nls_file
        else:
            raise ValueError(
                "`priors` kwarg must be path to valid prior model or `List[_Prior]`"
            )

        super(GeneralCarbonAlphaDataset, self).__init__(
            root, transform, pre_transform, pre_filter
        )
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.prior_model = torch.load(self.processed_paths[3])
        self.topologies = {self.mol_name: torch.load(self.processed_paths[2])}

    """
    def download(self):
        pass

    @property
    def raw_file_names(self):
        return [i[0] for i in self.raw_data_fns] + [j[1] for j in self.raw_data_fns]
    """

    @property
    def processed_file_names(self):
        return [
            "{}.pt".format(self.mol_name),
            "{}.pdb".format(self.mol_name),
            "{}_topologies.pt".format(self.mol_name),
            "priors.pt",
        ]

    def build_topos(self) -> Tuple[Topology, md.Trajectory]:
        """Method for preparing and sanitizing CG MLCG `Topology` and
        all-atom MDTraj `Trajectory` objects.

        Returns
        -------
        cg_topo:
            CG MLCG `Topology` object, imbued with sequential carbon alpha
            bonds, angles, and dihedrals
        traj:
            All-atom MDTraj `Trajectory` instance that contains protein atoms
        """
        traj = md.load(self.pdb_file).remove_solvent()
        md_topo = traj.topology
        topology = Topology.from_mdtraj(md_topo)
        cg_topo = Topology()

        for atom_type, at in zip(self.atom_types, list(md_topo.atoms)):
            cg_topo.add_atom(atom_type, "CA", at.residue.name, at.residue.index)
        cg_topo.names[0] += "-terminal"
        cg_topo.names[-1] += "-terminal"

        add_chain_bonds(cg_topo)
        add_chain_angles(cg_topo)
        add_chain_dihedrals(cg_topo)

        return cg_topo, traj

    def build_prior_nls(self, mlcg_topo: Topology) -> Dict:
        """Returns a neigborlist dictionary for assembling `AtomicData` objects.
        If the `GeneralCarbonAlphaDataset` has been instanced with a pre-existing
        prior and neighborlist, that neighborlist is loaded and returned. Otherwise,
        a neighborlist is constructed from scratch using the values of `self.priors`,
        `self.non_bond_cut`.

        Parameters
        ----------
        mlcg_topo:
            `mlcg.geometry.topology.Topology` instance for the CG molecule

        Returns
        -------
        prior_nls:
            Dictionary of `torch` neighborlists based on the priors used for
            producing the delta forces for the dataset
        """

        if self.prior_nls_file != None:
            with open(self.prior_nls_file, "rb") as nfile:
                prior_nls = pickle.load(nfile)
        else:
            prior_nls = {}
            for prior_class in self.prior_classes:
                prior_nls.update(**prior_class.neighbor_list(mlcg_topo))

            if any(
                [
                    isinstance(prior_class, Repulsion)
                    for prior_class in self.prior_classes
                ]
            ):
                # Construct non-bonded set based on self.non_bond_cut

                pairs = list(combinations(np.arange(len(self.atom_types)), 2))
                non_bonded_set = []
                for pair in pairs:
                    if abs(pair[0] - pair[1]) > self.non_bond_cut:
                        non_bonded_set.append(pair)
                non_bonded_set = torch.tensor(np.array(non_bonded_set).T)

                repulsion_nls = make_neighbor_list(
                    "repulsion", 2, non_bonded_set
                )
                prior_nls["repulsion"] = repulsion_nls

        return prior_nls

    def build_cg_data_list(
        self, traj: md.Trajectory, prior_nls: Dict
    ) -> Tuple[List[AtomicData], int]:
        """Method for assembling the CG `List[AtomicData]` from a list of coord/force filenames.
        If `self.exclude_cis_omega=True`, trajectories containing cis omega angles are discarded.
        If `self.pre_transform` is specified, it is applied to each element in the data list before
        returning.

        Parameters
        ----------
        traj:
            MDTraj `Trajectory` instance of the all-atom protein system. Assumed to be single-chain,
            and solvent/ion-free
        prior_nls:
            Dictionary of prior neighborlists

        Returns
        -------
        data_list:
            Uncollated list of `AtomicData` before prior force subtraction, complete with coords,
            forces, atom_types, (re-scaled) masses, and prior neighborlists.
        total_frames:
            The final number of CG-mapped frames in the CG dataset
        """
        data_list = []
        for i, data_files in enumerate(
            tqdm(
                self.raw_data_fns,
                desc="Producing CG dataset",
                disable=(not self.verbose),
            )
        ):
            aa_coords, aa_forces = self.coord_force_loader(*data_files)

            if self.exclude_cis_omega:
                if GeneralCarbonAlphaDataset.check_cis(traj, aa_coords) == True:
                    if self.verbose:
                        print(
                            "cis conformation detected in AA trajectory {}. Removing from dataset ...".format(
                                i
                            )
                        )
                    continue

            cg_coords = (self.coordmap @ aa_coords).astype(self.precision)
            cg_forces = (self.forcemap @ aa_forces).astype(self.precision)
            n_frames = cg_coords.shape[0]

            for i_frame in range(n_frames):
                pos = torch.from_numpy(
                    cg_coords[i_frame].reshape(self.num_cg_atoms, 3)
                )
                z = torch.from_numpy(self.atom_types)
                force = torch.from_numpy(
                    cg_forces[i_frame].reshape(self.num_cg_atoms, 3)
                )
                masses = torch.from_numpy(self.masses)
                data = AtomicData.from_points(
                    atom_types=z,
                    pos=pos,
                    forces=force,
                    masses=masses,
                    neighborlist=prior_nls,
                    name=self.mol_name,
                )

                if self.pre_filter != None and not self.pre_filter(data):
                    continue
                if self.pre_transform != None:
                    data = self.pre_transform(data)
                data_list.append(data)

        # additional general filter, if specified
        if self.pre_filter != None:
            data_list = list(filter(self.pre_filter, data_list))

        total_frames = len(data_list)

        return data_list, total_frames

    def fit_prior_model(
        self, data_for_prior_fit: List[AtomicData]
    ) -> torch.nn.Module:
        """Method for parametrizing priors from a subset of reference data

        Parameters
        ----------
        data_for_prior_fit:
            List of AtomicData

        Returns
        -------
        prior_model:
           torch.nn.Module of fitted prior terms
        """

        sub_datas, _, _ = collate(
            data_for_prior_fit[0].__class__,
            data_list=data_for_prior_fit,
            increment=True,
            add_batch=True,
        )

        prior_model, self.statistics = fit_baseline_models(
            sub_datas, self.beta, self.prior_classes
        )

        if self.dihedral_fit_kwargs != None:
            ### Override dihedral fits
            dihedral_dict = compute_statistics(
                sub_datas,
                beta=self.beta,
                target="dihedrals",
                TargetPrior=Dihedral,
                target_fit_kwargs=self.dihedral_fit_kwargs,
            )
            dihedral_prior = Dihedral(
                dihedral_dict, n_degs=self.dihedral_fit_kwargs["n_degs"]
            )
            prior_model["dihedrals"] = dihedral_prior

        for k in prior_model.keys():
            prior_model[k] = GradientsOut(prior_model[k], targets="forces")

        return prior_model

    @staticmethod
    def check_cis(traj: md.Trajectory, coords: np.ndarray) -> bool:
        """Checks an all-atom trajectory for cis conformations (as measured by low CA-CA bond distances)
        and flags the trajectory if ANY are found. Units of ANGSTROMS are assumed

        Parameters
        ----------
        traj:
            all-atom mdtraj.Trajecory instance
        coords:
            numpy array of all-atom cartesian trajectory coordinates of shape (n_frames, n_atoms, 3)

        Returns
        -------
        has_cis:
            if True, the input all-atom trajectory contains one or more cis-conformations
        """

        has_cis = False
        traj.xyz = coords
        traj.time = np.arange(len(coords))
        ca_idx = traj.topology.select("name CA")
        ca_traj = traj.atom_slice(ca_idx)
        ca_atom_pairs = list(combinations(np.arange(len(ca_idx)), 2))
        distances = md.compute_distances(ca_traj, ca_atom_pairs, periodic=False)
        for i in range(len(ca_atom_pairs)):
            pair = ca_atom_pairs[i]
            if abs(pair[1] - pair[0]) == 1:
                if any(distances[:, i] < 3.1):
                    has_cis = True
        return has_cis

    def produce_delta_forces(
        self, data_list: AtomicData, prior_model: torch.nn.Module
    ) -> AtomicData:
        """Method for subtracting prior forces from mapped CG forces to produce a delta force
        dataset

        Parameters
        ----------
        data_list:
            List of `AtomicData` containing the mapped CG coordinates and forces
        prior_model:
            `torch.nn.Module` of prior terms which together comprise the prior model

        Returns
        -------
        delta_data:
            collated `AtomicData` wherein the original mapped CG-mapped forces have
            been transformed to delta forces by subtracting prior forces. The prior forces
        """

        # Here, the data_list is modified in-place over several chunks
        # Iterating over a sequence does not implicitly make a copy in Python

        chunks = tuple(chunker(data_list, self.delta_force_batch_size))
        for sub_data_list in tqdm(
            chunks, desc="Producing delta forces", disable=(not self.verbose)
        ):
            _ = remove_baseline_forces(
                sub_data_list,
                prior_model,
            )

        datas, slices = self.collate(data_list)

        # remove baseline_forces and prior's neighbor lists to reduce
        # memory footprint of the dataset
        delattr(datas, "baseline_forces")
        datas.neighbor_list = {}

        return datas, slices

    def save_h5_dataset(
        self, coords: np.array, delta_forces: np.array, atom_types: np.array
    ):
        """Method for saving an h5 dataset containing the coordinates, CG atom types, and

        Parameters
        ----------
        coords:
            `np.array` of CG coordinates
        delta_forces:
            `np.array` of CG delta forces
        atom_types:
            `np.array` of CG atom types
        """

        with h5py.File(
            join(
                self.root,
                "processed",
                "{}_delta_dataset.h5".format(self.mol_name),
            ),
            "w",
        ) as f:
            metaset = f.create_group(self.mol_name.upper())
            hdf_group = metaset.create_group(self.mol_name)
            hdf_group.create_dataset("cg_coords", data=coords)
            hdf_group.create_dataset("cg_delta_forces", data=delta_forces)
            hdf_group.attrs["cg_embeds"] = atom_types
            hdf_group.attrs["N_frames"] = coords.shape[0]

    def process(self):
        """Main pipeline for producing the CG dataset and fitted priors"""

        # Build topology-related objects
        mlcg_topo, traj = self.build_topos()
        torch.save((mlcg_topo), self.processed_paths[2])

        # Build neighborlists
        prior_nls = self.build_prior_nls(mlcg_topo)

        # Process CG data
        data_list, n_frames = self.build_cg_data_list(traj, prior_nls)

        # Save some starting configurations for later simulations
        random_sim_idx = np.random.choice(
            np.arange(len(data_list)), self.num_sim_starts, replace=False
        )
        frames_for_sims = []
        for ri in random_sim_idx:
            frames_for_sims.append(data_list[ri])
        torch.save(
            frames_for_sims, join(self.root, "processed", "sim_starts.pt")
        )

        # Get prior model
        if self.prior_file != None and self.prior_nls_file != None:
            prior_model = torch.load(self.prior_file)
        else:
            # Fit prior model
            random_prior_idx = np.random.choice(
                np.arange(len(data_list)), self.num_prior_samples, replace=False
            )
            data_for_prior_fit = [data_list[idx] for idx in random_prior_idx]
            prior_model = self.fit_prior_model(data_for_prior_fit)

        torch.save(prior_model, self.processed_paths[3])

        # Produce delta forces
        datas, slices = self.produce_delta_forces(data_list, prior_model)

        # Check for poor prior choices
        if any(torch.abs(datas.forces).flatten() > self.delta_check_threshold):
            warnings.warn("Large delta forces detected in dataset")

        torch.save((datas, slices), self.processed_paths[0])

        # Save H5 dataset
        self.save_h5_dataset(
            (datas.pos.numpy()).reshape(n_frames, self.num_cg_atoms, 3),
            (datas.forces.numpy()).reshape(n_frames, self.num_cg_atoms, 3),
            self.atom_types,
        )
