from typing import List, Tuple
import pytest
import numpy as np
import torch
import mdtraj as md
import tempfile
import os.path as osp
from os.path import join
from mdtraj.core.trajectory import Trajectory
from mlcg.geometry.topology import Topology
from mlcg.data import AtomicData
from .general_datasets import GeneralCarbonAlphaDataset
import os


class MockProteinReference(object):
    """Simple pentapeptide (LPGWK) reference dataset to test `GeneralCarbonAlphaDataset`.

    Provides the following (mock) objects:

    1. All-atom coordinates and forces (filenames and numpys)
    2. A loader function
    3. All-atom PDB file
    4. CG Masses
    5. CG Atom types
    6. CG Maps (both slice for coords and forces)
    """

    def __init__(self, temp_dir: str):
        os.mkdir(join(temp_dir, "raw"))
        self.masses = np.array([12.0 for i in range(5)]) / 418.3
        self.atom_types = np.arange(1, 6)
        self.coord_files = [
            join(temp_dir, "raw", "aa_coords_{}.npy".format(i))
            for i in range(5)
        ]
        self.force_files = [
            join(temp_dir, "raw", "aa_forces_{}.npy".format(i))
            for i in range(5)
        ]
        self.pdb_file = join(temp_dir, "raw", "test_peptide.pdb")

        with open(join(temp_dir, self.pdb_file), "w") as pfile:
            pdb_str = MockProteinReference.gen_pdb()
            pfile.write(pdb_str)

        traj = md.load(join(temp_dir, self.pdb_file))
        ca_idx = traj.topology.select("name CA")
        slice_map = np.zeros((5, traj.xyz.shape[1]))
        for row, idx in enumerate(ca_idx):
            slice_map[row][idx] = 1
        self.coordmap = slice_map
        self.forcemap = slice_map
        coord_force_sets = MockProteinReference.generate_coords_and_forces(traj)
        for cf_set, cfile, ffile in zip(
            coord_force_sets, self.coord_files, self.force_files
        ):
            np.save(cfile, cf_set[0])
            np.save(ffile, cf_set[1])

    @staticmethod
    def coord_force_loader(coord_fn: str, force_fn: str):
        coords = np.load(coord_fn)
        forces = np.load(force_fn)
        return coords, forces

    @staticmethod
    def generate_coords_and_forces(
        traj: Trajectory,
    ) -> List[Tuple[np.array, np.array]]:
        num_samples = 10000
        # convert starting coordinates to angstroms
        init_aa_coords = 10.0 * traj.xyz

        # create 4 noisey mock coordinate and force sets
        coord_force_sets = []
        for i in range(5):
            sub_coords = np.repeat(init_aa_coords, num_samples, axis=0)
            sub_forces = np.repeat(
                init_aa_coords, num_samples, axis=0
            )  # Here we just take the forces as coords
            coord_noise = np.random.normal(0, scale=0.01, size=sub_coords.shape)
            force_noise = np.random.normal(0, scale=0.01, size=sub_coords.shape)
            coord_force_sets.append(
                (sub_coords + coord_noise, sub_forces + force_noise)
            )
        return coord_force_sets

    @staticmethod
    def gen_pdb():
        pdb_str = (
            "SEQRES   1 A    5  LEU PRO GLY TRP LYS\n"
            "ATOM      1  N   LEU A   1      -3.236  -2.406  -1.437  1.00  0.00           N\n"
            "ATOM      2  CA  LEU A   1      -1.787  -2.406  -1.437  1.00  0.00           C\n"
            "ATOM      3  C   LEU A   1      -1.264  -0.977  -1.437  1.00  0.00           C\n"
            "ATOM      4  O   LEU A   1      -0.371  -0.642  -0.661  1.00  0.00           O\n"
            "ATOM      5  CB  LEU A   1      -1.235  -3.107  -2.669  1.00  0.00           C\n"
            "ATOM      6  CG  LEU A   1       0.246  -3.405  -2.459  1.00  0.00           C\n"
            "ATOM      7  CD1 LEU A   1       0.746  -4.309  -3.581  1.00  0.00           C\n"
            "ATOM      8  CD2 LEU A   1       1.034  -2.099  -2.469  1.00  0.00           C\n"
            "ATOM      9  N   PRO A   2      -1.823  -0.136  -2.310  1.00  0.00           N\n"
            "ATOM     10  CA  PRO A   2      -1.412   1.250  -2.408  1.00  0.00           C\n"
            "ATOM     11  C   PRO A   2      -1.567   1.938  -1.058  1.00  0.00           C\n"
            "ATOM     12  O   PRO A   2      -0.660   2.634  -0.608  1.00  0.00           O\n"
            "ATOM     13  CB  PRO A   2      -2.142   1.756  -3.601  1.00  0.00           C\n"
            "ATOM     14  CG  PRO A   2      -2.384   0.599  -4.541  1.00  0.00           C\n"
            "ATOM     15  CD  PRO A   2      -2.710  -0.579  -3.671  1.00  0.00           C\n"
            "ATOM     16  N   GLY A   3      -2.719   1.740  -0.415  1.00  0.00           N\n"
            "ATOM     17  CA  GLY A   3      -2.988   2.339   0.877  1.00  0.00           C\n"
            "ATOM     18  C   GLY A   3      -1.915   1.931   1.876  1.00  0.00           C\n"
            "ATOM     19  O   GLY A   3      -1.384   2.773   2.597  1.00  0.00           O\n"
            "ATOM     20  N   TRP A   4      -1.597   0.635   1.916  1.00  0.00           N\n"
            "ATOM     21  CA  TRP A   4      -0.591   0.121   2.824  1.00  0.00           C\n"
            "ATOM     22  C   TRP A   4       0.733   0.835   2.595  1.00  0.00           C\n"
            "ATOM     23  O   TRP A   4       1.380   1.265   3.547  1.00  0.00           O\n"
            "ATOM     24  CB  TRP A   4      -0.365  -1.367   2.610  1.00  0.00           C\n"
            "ATOM     25  CG  TRP A   4       0.689  -1.984   3.499  1.00  0.00           C\n"
            "ATOM     26  CD1 TRP A   4       0.520  -2.454   4.742  1.00  0.00           C\n"
            "ATOM     27  CD2 TRP A   4       2.043  -2.187   3.203  1.00  0.00           C\n"
            "ATOM     28  CE2 TRP A   4       2.670  -2.758   4.227  1.00  0.00           C\n"
            "ATOM     29  CE3 TRP A   4       2.755  -1.880   2.037  1.00  0.00           C\n"
            "ATOM     30  CZ3 TRP A   4       4.128  -2.202   2.022  1.00  0.00           C\n"
            "ATOM     31  CH2 TRP A   4       4.738  -2.781   3.077  1.00  0.00           C\n"
            "ATOM     32  CZ2 TRP A   4       4.031  -3.087   4.235  1.00  0.00           C\n"
            "ATOM     33  NE1 TRP A   4       1.792  -2.939   5.182  1.00  0.00           N\n"
            "ATOM     34  N   LYS A   5       1.135   0.958   1.328  1.00  0.00           N\n"
            "ATOM     35  CA  LYS A   5       2.377   1.617   0.979  1.00  0.00           C\n"
            "ATOM     36  C   LYS A   5       2.388   3.038   1.526  1.00  0.00           C\n"
            "ATOM     37  O   LYS A   5       3.402   3.401   2.117  1.00  0.00           O\n"
            "ATOM     38  OXT LYS A   5       1.380   3.716   1.337  1.00  0.00           O\n"
            "ATOM     39  CB  LYS A   5       2.556   1.695  -0.529  1.00  0.00           C\n"
            "ATOM     40  CG  LYS A   5       2.765   0.291  -1.086  1.00  0.00           C\n"
            "ATOM     41  CD  LYS A   5       2.963   0.367  -2.596  1.00  0.00           C\n"
            "ATOM     42  CE  LYS A   5       3.158  -1.039  -3.154  1.00  0.00           C\n"
            "ATOM     43  NZ  LYS A   5       3.337  -0.966  -4.612  1.00  0.00           N\n"
            "END"
        )

        return pdb_str


def test_general_dataset_run():
    """Tests to make sure that GeneralCarbonAlphaDataset processes a test
    tetrapeptide system properly (using default kwargs)
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        tetra_dataset = MockProteinReference(temp_dir)
        raw_data_fns = [
            (tetra_dataset.coord_files[i], tetra_dataset.force_files[i])
            for i in range(5)
        ]

        mlcg_dataset = GeneralCarbonAlphaDataset(
            temp_dir,
            tetra_dataset.coordmap,
            tetra_dataset.forcemap,
            raw_data_fns,
            tetra_dataset.atom_types,
            tetra_dataset.masses,
            tetra_dataset.pdb_file,
            tetra_dataset.coord_force_loader,
            num_prior_samples=10000,
        )
