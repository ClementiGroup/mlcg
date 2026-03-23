import os
from os.path import join

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, extract_tar
from torch_geometric.data.collate import collate
from shutil import copy
import mdtraj

from aggforce import (
    LinearMap,
    guess_pairwise_constraints,
    project_forces,
    constraint_aware_uni_map,
    qp_linear_map,
)


from ..utils import tqdm, download_url
from ..geometry.topology import Topology
from ..geometry.statistics import fit_baseline_models
from ..cg import build_cg_matrix, build_cg_topology, CA_MAP
from ..data import AtomicData
from ..nn import (
    HarmonicBonds,
    HarmonicAngles,
    Repulsion,
    Dihedral,
    GradientsOut,
)
from .utils import remove_baseline_forces, chunker


class ChignolinDataset(InMemoryDataset):
    r"""Dataset for training a CG model of the chignolin protein following a Cα CG mapping using the all-atom data from [CGnet]_.

    This Dataset produces delta forces for model training, in which the CG prior forces (harmonic bonds and angles and a repulsive term) have been subtracted from the full CG forces.
    """

    #:Temperature used to generate the underlying all-atom data in [K]
    temperature = 350  # K
    #:Boltzmann constan in kcal/mol/K
    kB = 0.0019872041
    #:
    _priors_cls = [HarmonicBonds, HarmonicAngles, Repulsion]

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        mapping: str = "slice_aggregate",
        terminal_embeds: bool = False,
        max_num_files: int = None,
    ):
        self.terminal_embeds = terminal_embeds
        self.priors_cls = self._priors_cls
        self.mapping = mapping
        self.beta = 1 / (self.temperature * self.kB)
        if max_num_files is None:
            self.max_num_files = 100000
        else:
            self.max_num_files = max_num_files
        super(ChignolinDataset, self).__init__(
            root, transform, pre_transform, pre_filter
        )
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.prior_models = torch.load(self.processed_paths[3])
        self.topologies = {"cln": torch.load(self.processed_paths[2])}

    def download(self):
        # Download to `self.raw_dir`.
        host_link = (
            "http://ftp.mi.fu-berlin.de/pub/cmb-data/chignolin_trajectories/"
        )
        url_forces = f"{host_link}/chignolin_forces_nowater.tar.gz"
        url_coords = f"{host_link}/chignolin_coords_nowater.tar.gz"
        url_inputs = f"{host_link}/chignolin_generators.tar.gz"
        path_inputs = download_url(url_inputs, self.raw_dir)
        path_coord = download_url(url_coords, self.raw_dir)
        path_forces = download_url(url_forces, self.raw_dir)

        # extract files
        for fn in tqdm(self.raw_paths, desc="Extracting archives"):
            extract_tar(fn, self.raw_dir, mode="r:gz")

    @property
    def raw_file_names(self):
        return [
            "chignolin_generators.tar.gz",
            "chignolin_forces_nowater.tar.gz",
            "chignolin_coords_nowater.tar.gz",
        ]

    @property
    def processed_file_names(self):
        return ["chignolin.pt", "chignolin.pdb", "topologies.pt", "priors.pt"]

    @staticmethod
    def get_data_filenames(coord_dir, force_dir):
        tags = [
            os.path.basename(fn).replace("chig_coor_", "").replace(".npy", "")
            for fn in os.listdir(coord_dir)
        ]

        coord_fns = {}
        for tag in tags:
            fn = f"chig_coor_{tag}.npy"
            coord_fns[tag] = join(coord_dir, fn)
        forces_fns = {}
        for tag in tags:
            fn = f"chig_force_{tag}.npy"
            forces_fns[tag] = join(force_dir, fn)
        return coord_fns, forces_fns

    def process(self):
        coord_dir = join(self.raw_dir, "coords_nowater")
        force_dir = join(self.raw_dir, "forces_nowater")

        topology_fn = join(self.raw_dir, "chignolin_50ns_0/structure.pdb")
        topo = mdtraj.load(topology_fn).remove_solvent().topology
        topology = Topology.from_mdtraj(topo)
        embeddings, masses, cg_matrix, _ = build_cg_matrix(
            topology, cg_mapping=CA_MAP, special_terminal=self.terminal_embeds
        )
        cg_topo = build_cg_topology(
            topology, cg_mapping=CA_MAP, special_terminal=self.terminal_embeds
        )
        copy(topology_fn, self.processed_paths[1])
        prior_nls = {}
        for cls in self.priors_cls:
            prior_nls.update(**cls.neighbor_list(cg_topo))

        n_beads = cg_matrix.shape[0]
        embeddings = np.array(embeddings, dtype=np.int64)

        coord_fns, forces_fns = self.get_data_filenames(coord_dir, force_dir)

        coords = np.load(next(iter(coord_fns.values())))
        forces = np.load(next(iter(forces_fns.values())))

        config_map = LinearMap(cg_matrix)
        cg_matrix = config_map.standard_matrix
        # taking only first 100 frames gives same results in ~1/15th of time
        constraints = guess_pairwise_constraints(coords[:], threshold=5e-3)
        if self.mapping == "slice_aggregate":
            method = constraint_aware_uni_map
            force_agg_results = project_forces(
                coords=coords,
                forces=forces,
                coord_map=config_map,
                constrained_inds=constraints,
                method=method,
            )
        elif self.mapping == "slice_optimize":
            method = qp_linear_map
            l2 = 1e3
            force_agg_results = project_forces(
                coords=coords,
                forces=forces,
                coord_map=config_map,
                constrained_inds=constraints,
                method=method,
                l2_regularization=l2,
            )
        else:
            raise RuntimeError(
                f"Force mapping {self.mapping} is neither 'slice_aggregate' nor 'slice_optimize'."
            )
        f_proj = force_agg_results["tmap"].force_map.standard_matrix

        data_list = []
        ii_frame = 0
        file_counter = 0
        for i_traj, tag in enumerate(tqdm(coord_fns, desc="Load Dataset")):
            forces = np.load(forces_fns[tag])
            cg_forces = np.array(
                np.einsum("mn, ind-> imd", f_proj, forces), dtype=np.float32
            )
            coords = np.load(coord_fns[tag])
            cg_coords = np.array(
                np.einsum("mn, ind-> imd", cg_matrix, coords), dtype=np.float32
            )

            n_frames = cg_coords.shape[0]
            file_counter += 1
            if file_counter > self.max_num_files:
                break
            for i_frame in range(n_frames):
                pos = torch.from_numpy(cg_coords[i_frame].reshape(n_beads, 3))
                z = torch.from_numpy(embeddings)
                force = torch.from_numpy(cg_forces[i_frame].reshape(n_beads, 3))

                data = AtomicData.from_points(
                    atom_types=z,
                    pos=pos,
                    forces=force,
                    masses=masses,
                    neighborlist=prior_nls,
                    traj_id=i_traj,
                    name="cln",
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

        torch.save((cg_topo), self.processed_paths[2])
        torch.save(baseline_models, self.processed_paths[3])
        torch.save((datas, slices), self.processed_paths[0])


class ChignolinDatasetWithDihedralPriors(ChignolinDataset):
    """
    Inherit from Chignolin Dataset and redefine priors to compute
    Modify _prior_cls to include priors to compute
    Must provide
        -----
    """

    _priors_cls = [HarmonicBonds, HarmonicAngles, Repulsion, Dihedral]

    def __init__(
        self, root, transform=None, pre_transform=None, pre_filter=None
    ):
        self.priors_cls = self._priors_cls
        super(ChignolinDatasetWithDihedralPriors, self).__init__(
            root, transform, pre_transform, pre_filter
        )
