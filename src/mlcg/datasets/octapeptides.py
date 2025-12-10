import os
from os.path import join

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, extract_tar
from torch_geometric.data.collate import collate
from shutil import copy
import mdtraj


from ..utils import tqdm, download_url
from ..geometry.topology import Topology
from ..geometry.statistics import fit_baseline_models
from ..cg import (
    build_cg_matrix,
    build_cg_topology,
    OPEPS_MAP,
    swap_mapping_rows,
)
from ..data import AtomicData
from ..nn import (
    HarmonicBonds,
    HarmonicAngles,
    Repulsion,
    GradientsOut,
)
from .utils import remove_baseline_forces, chunker


class OctapeptidesDataset(InMemoryDataset):
    """Octapeptides dataset"""

    #:Temperature used to generate the underlying all-atom data in [K]
    temperature = 300  # K
    #:Boltzmann constan in kcal/mol/K
    kB = 0.0019872041

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        isolate_termini_stats=True,
    ):
        self.beta = 1 / (self.temperature * self.kB)
        super(OctapeptidesDataset, self).__init__(
            root, transform, pre_transform, pre_filter
        )

        self.peptide_nums = np.arange(200)
        self.isolate_termini_stats
        self.pseudobonds = pseudobonds
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.prior_models = torch.load(self.processed_paths[3])
        self.topologies = torch.load(self.processed_paths[2])

    def download(self):
        # Download to `self.raw_dir`.
        url_topologies = [
            "http://pub.htmd.org/largepeptides_simulations.tar.gz",
            "http://pub.htmd.org/octapeptides_2_structures.tar.gz",
        ]
        url_forces = [
            "http://pub.htmd.org/largepeptides_1_forces_nowater.tar.gz",
            "http://pub.htmd.org/largepeptides_2_forces_nowater.tar.gz",
        ]
        url_coords = [
            "http://pub.htmd.org/largepeptides_1_coords_nowater.tar.gz",
            "http://pub.htmd.org/largepeptides_2_coords_nowater.tar.gz",
        ]
        if self.reduced:
            url_topologies = url_topologies[0]
            url_forces = url_forces[0]
            url_coords = url_coords[0]

        path_topologies = download_url(url_topologies, self.raw_dir)
        path_coord = download_url(url_coords, self.raw_dir)
        path_forces = download_url(url_forces, self.raw_dir)

        # extract files
        for fn in tqdm(self.raw_paths, desc="Extracting archives"):
            extract_tar(fn, self.raw_dir, mode="r:gz")

    @property
    def raw_file_names(self):
        return [
            "topo",
            "largepeptides_1_forces_nowater.tar.gz",
            "largepeptides_1_coords_nowater.tar.gz",
            "topo",
            "largepeptides_2_forces_nowater.tar.gz",
            "largepeptides_2_coords_nowater.tar.gz",
        ]

    @property
    def processed_file_names(self):
        # return ["chignolin.pt", "chignolin.pdb", "topologies.pt", "priors.pt"]
        pass

    @staticmethod
    def get_data_filenames():
        topo_directory_batches = np.linspace(200, 1100, 16)
        topo_directories = {i: "batch_" + str(i) for i in range(15)}
        coord_files = {}
        force_files = {}
        topo_files = {}
        for num in peptide_nums:
            if num < 200:
                coord_fns = sorted(
                    glob(self.raw_dir + "/coords_nowater/opep_{:04d}/*.npy")
                )
                force_fns = sorted(
                    glob(self.raw_dir + "/forces_nowater/opep_{:04d}/*.npy")
                )
                topo_fn = (
                    self.raw_dir + "/largepeptides/opep_{:0d}/filtered.pdb"
                )
            else:
                coord_fns = sorted(
                    glob(self.raw_dir + "/coords_nowater/opep_{:04d}/*.npy")
                )
                force_fns = sorted(
                    glob(self.raw_dir + "/forces_nowater/opep_{:04d}/*.npy")
                )
                batch = np.digitize(num, topo_directory_batches, right=True)
                subdir = topo_directories[batch]
                topo_fn = (
                    self.raw_dir
                    + "/largepeptides2/{}/opep_{:0d}/opep_0200/input/e1s1_opep_0200/filtered.pdb".format(
                        subdir
                    )
                )
            topo_files[num] = topo_fn
            coord_files[num] = coord_fns
            force_files[num] = force_fns

        return coord_files, forces_files, topo_files

    def assemble_neighbor_lists(self, cg_topo, connectivity_matrix):
        """Helper function for assembling neighbor lists"""

        cg_atom_names = [name.split("_")[1] for name in cg_topo.names]
        bonds = []
        bond_tags = []
        angles = []
        angle_tags = []

        bond_edges = cg_topo.bonds2torch()
        angle_edges = cg_topo.angles2torch()

        if self.isolate_termini_stats:
            n_term_atoms = []
            for i, atom in enumerate(cg_atom_names):
                if atom.name == "O":
                    n_term_atoms.append(i)
                    break
                else:
                    n_term_atoms.append(i)
            c_term_atoms = []
            for i, atom in enumerate(reversed(cg_atom_names)):
                if atom.name == "N":
                    c_term_atoms.append(len(atoms) - i)
                    break
                else:
                    c_term_atoms.append(len(atoms) - i)
            c_term_atoms = sorted(c_term_atoms)

            n_term_bonds, c_term_bonds, bulk_bonds = isolate_features(
                [n_term_atoms, c_term_atoms], bond_edges
            )
            n_term_angles, c_term_angles, bulk_angles = isolate_features(
                [n_term_atoms, c_term_atoms], angles_edges
            )

            bonds.extend((n_term_bonds, bulk_bonds, c_term_bonds))
            bond_tags.extend(("n_term_bonds", "bulk_bonds", "c_term_bonds"))
            angles.extend((n_term_angles, bulk_angles, c_term_angles))
            angles.extend(("n_term_angles", "bulk_angles", "c_term_angles"))
        else:
            bonds.append(bond_edges)
            bond_tags.append("bonds")
            angles.append(angle_edges)
            angle_tags.append("angles")

        non_bonded_edges = make_nonbonded_set(
            cg_topo,
            minimum_separation=4,
            residue_indices=residue_indices,
            residue_exclusion=True,
        )

        prior_nls = {}
        for tag, edge_list in zip(bond_tags, bonds):
            prior_nls[tag] = make_neighbor_list(tag, 2, edge_list)

        for tag, edge_list in zip(angle_tags, angles):
            prior_nls[tag] = make_neighbor_list(tag, 3, edge_list)

        prior_nls["non_bonded"] = make_neighbor_list(
            "non_bonded", 2, non_bonded_edges
        )

        return prior_nls

    def process(self):
        topo_fns, coord_fns, force_fns = self.get_data_filenames()
        for num in peptide_nums:
            coord_fn_list = coord_fns[num]
            force_fn_list = force_fns[num]
            assert len(coord_fn_list) == len(force_fn_list)

            topo = mdtraj.load(topo_fns[num]).remove_solvent().topology
            topology = Topology.from_mdtraj(topo)

            types, masses, cg_matrix, mapping = build_cg_matrix(
                topology, cg_mapping=OPEPS_MAP
            )

            # swap CA and CB for all PRO residues in order to have
            # a consistent per residue mapping of (N,CA,CB,C,O)

            s_types, s_masses, s_cg_matrix, s_mapping = swap_mapping_rows(
                topology, "PRO", ["CA", "CB"], types, masses, cg_matrix, mapping
            )

            cg_topo = build_cg_topology(
                topology,
                s_mapping,
                special_terminal=False,
            )
            resnames = [res.name for res in list(topo.residues)]
            connectivity_matrix = build_opeps_connectivity_matrix(resnames)

            # add bonds and angles according to the connectivity matrix
            make_general_bonds(cg_topo, connectivity_matrix)
            make_general_angles(cg_topo, connectivity_matrix)

            torch.save((cg_topo), self.processed_paths[2])
            # make the neighborlists for bonds, angles, pseudobonds,
            # and non-bonded pairs

            prior_nls = self.assemble_neighbor_lists(
                cg_topo, connectivity_matrix
            )

            n_beads = cg_matrix.shape[0]

            f_proj = np.dot(
                np.linalg.inv(np.dot(s_cg_matrix, s_cg_matrix.T)), s_cg_matrix
            )

            data_list = []
            ii_frame = 0
            for i_traj, (coord_fn, force_fn) in enumerate(
                tqdm(
                    zip(coord_fns, force_fns),
                    total=len(coord_fns),
                    desc="Load Dataset",
                )
            ):
                forces = np.load(force_fn)
                cg_forces = np.array(
                    np.einsum("mn, ind-> imd", f_proj, forces), dtype=np.float32
                )

                coords = np.load(coord_fn)
                cg_coords = np.array(
                    np.einsum("mn, ind-> imd", cg_matrix, coords),
                    dtype=np.float32,
                )

                n_frames = cg_coords.shape[0]

                for i_frame in range(n_frames):
                    pos = torch.from_numpy(
                        cg_coords[i_frame].reshape(n_beads, 3)
                    )
                    z = cg_topo.types2torch()
                    force = torch.from_numpy(
                        cg_forces[i_frame].reshape(n_beads, 3)
                    )
                    masses = torch.tensor(s_masses)

                    data = AtomicData.from_points(
                        atom_types=z,
                        pos=pos,
                        forces=force,
                        masses=masses,
                        neighborlist=prior_nls,
                        mol_id=num,
                        traj_id=i_traj,
                        frame_id=i_frame,
                    )

                    if self.pre_filter != None and not self.pre_filter(data):
                        continue
                    if self.pre_transform != None:
                        data = self.pre_transform(data)
                    data_list.append(data)
                    ii_frame += 1

        print("collating data_list")
        datas, _, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=True,
            add_batch=True,
        )

        print("fitting baseline models")
        baseline_models, self.statistics = fit_baseline_models(
            datas, self.beta, self.priors_cls
        )

        for k in baseline_models.keys():
            baseline_models[k] = GradientsOut(
                baseline_models[k], targets="forces"
            )

        batch_size = 256
        chunks = tuple(chunker(data_list, batch_size))
        for sub_data_list in tqdm(chunks, "Removing baseline forces"):
            _ = remove_baseline_forces(
                sub_data_list,
                baseline_models,
            )

        print("collating data_list")
        datas, slices = self.collate(data_list)

        # remove baseline_forces and prior's neighbor lists to reduce
        # memory footprint of the dataset
        delattr(datas, "baseline_forces")
        datas.neighbor_list = {}

        torch.save(baseline_models, self.processed_paths[3])
        torch.save((datas, slices), self.processed_paths[0])
