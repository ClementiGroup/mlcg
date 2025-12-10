import pytest
import numpy as np
import torch
import mdtraj as md
import tempfile
import os.path as osp
from ase.build import molecule

from mlcg.geometry.topology import Topology
from mlcg.datasets.utils import write_PDB
from mlcg.data import AtomicData


class MockDataset(object):
    """Mock Dataset object with minimum attributes necessary
    to test mlcg.datasets.utils.write_PDB
    """

    def __init__(self, data, topology_dict):
        self.data = data
        self.topologies = topology_dict


def test_mdtraj_dump():
    """Tests to make sure that write_PDB dumps output
    correctly when it is later loaded by MDTraj later"""

    with tempfile.TemporaryDirectory() as temp_dir:
        ala2_pdb = (
            "MODEL        0\n"
            "ATOM      5  C   ACE A   1       2.770  25.800   1.230  1.00  0.00           C\n"
            "ATOM      7  N   ALA A   2       3.270  24.640   0.690  1.00  0.00           N\n"
            "ATOM      9  CA  ALA A   2       2.480  23.690  -0.190  1.00  0.00           C\n"
            "ATOM     11  CB  ALA A   2       3.470  23.160  -1.270  1.00  0.00           C\n"
            "ATOM     15  C   ALA A   2       1.730  22.590   0.490  1.00  0.00           C\n"
            "ATOM     17  N   NME A   3       0.400  22.430   0.210  1.00  0.00           N\n"
            "TER      18      NME A   3\n"
            "ENDMDL\n"
            "CONECT    1    2\n"
            "CONECT    2    1\n"
            "CONECT    5    6\n"
            "CONECT    6    5\n"
            "END"
        )
        with open(osp.join(temp_dir, "test_pdb.pdb"), "w") as pfile:
            pfile.write(ala2_pdb)

        md_pdb = md.load(osp.join(temp_dir, "test_pdb.pdb"))
        for atom in md_pdb.topology.atoms:
            print(atom, atom.residue, atom.residue.index, atom.residue.resSeq)

        types = []
        names = []
        resids = []
        for i, atom in enumerate(md_pdb.topology.atoms):
            types.append(i + 1)
            names.append(atom.name)
            resids.append(atom.residue.name)
        mlcg_topo = Topology.from_mdtraj(md_pdb.topology)
        n_atoms = len(mlcg_topo.types)

        data = AtomicData.from_points(
            pos=torch.tensor(md_pdb.xyz.reshape(n_atoms, 3)),
            atom_types=torch.tensor(types),
        )
        data.name = ["ala2"]
        topology_dict = {"ala2": mlcg_topo}
        dataset = MockDataset(data, topology_dict)
        mlcg_coords = dataset.data.pos.detach().numpy().reshape(1, n_atoms, 3)
        write_PDB(dataset, fout=osp.join(temp_dir, "mlcg_pdb.pdb"))
        new_md_pdb = md.load(osp.join(temp_dir, "mlcg_pdb.pdb"))

        # check names
        assert all(
            [
                mlcg_name == atom.name
                for mlcg_name, atom in zip(
                    mlcg_topo.names, list(new_md_pdb.topology.atoms)
                )
            ]
        )
        # check resnames
        unique_resids, idx = np.unique(mlcg_topo.resids, return_index=True)
        unique_resnames = [mlcg_topo.resnames[i] for i in idx]
        assert all(
            [
                mlcg_name == res.name
                for mlcg_name, res in zip(
                    unique_resnames, list(new_md_pdb.topology.residues)
                )
            ]
        )
        # check resids
        assert all(
            [
                mlcg_id == atom.residue.resSeq
                for mlcg_id, atom in zip(
                    mlcg_topo.resids, list(new_md_pdb.topology.atoms)
                )
            ]
        )
        # check coordinates
        np.testing.assert_array_equal(mlcg_coords, md_pdb.xyz)
