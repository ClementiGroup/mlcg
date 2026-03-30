import torch
import grappa
from grappa.data import Molecule
from mlcg.data import AtomicData
from mlcg.nn.prior.harmonic import Harmonic, HarmonicAnglesRaw
from mlcg.nn.prior.fourier_series import FourierSeries
import pytorch_lightning as pl
import h5py
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from mlcg.nn.prior.harmonic import HarmonicImpropers

from grappa.models.grappa import GrappaModel
from grappa.utils.graph_utils import get_default_statistics
from typing import Union, List, Dict

# from mlcg.simulation import LangevinSimulation
# from mlcg.pl import merge_priors_and_checkpoint
# from time import ctime

# from deeptime.decomposition import TICA
# from deeptime.util import energy2d
# from deeptime.plots import plot_energy2d
# from deeptime.clustering import KMeans, ClusterModel
# from deeptime.markov.msm import MaximumLikelihoodMSM

# from glob import glob
# from itertools import combinations

# import matplotlib.colors as mplc
# import matplotlib.pyplot as plt
# import mdtraj as md
import numpy as np

# import os
import pickle as pkl
# from tqdm import tqdm
from typing import List

class MLCGGrappaWrapper(torch.nn.Module):
    """
    Wrapper to use Grappa with MLCG's AtomicData format, preserving gradients.

    Parameters
    ----------
    grappa_tag : str, default='latest'
        Which pretrained Grappa model to load
    """

    def __init__(self, grappa_tag: str = "latest"):
        super().__init__()
        self.grappa_model = grappa.Grappa.from_tag(grappa_tag)
        self.raw_model = self.grappa_model.model
        self.r_max = None  # Topology-based, no cutoff needed
        self.name = "grappa"  # PLModel uses this to find predictions
        self.derivative = True

        # Initialize topology attributes to None
        self.g = None
        self.bonds_i = None
        self.bonds_j = None
        self.angles_i = None
        self.angles_j = None
        self.angles_k = None
        self.dihedrals_mapping = None

    def _build_topology(self, atomic_data: AtomicData):
        """Build Grappa topology from first AtomicData sample (call once).
        
        Supports two data formats:
        1. Manual/notebook format: neighbor_list contains 'bonds', 'angles', 'pseudo_ca_dihedral' keys
        2. MLCG H5 format: topology built from 'exc_pairs' attribute (bond pairs)
        """
        n_atoms = atomic_data["atom_types"].size(0)
        
        # Check if neighbor_list has the expected structure (notebook format)
        nl = atomic_data.get("neighbor_list", {})
        if "bonds" in nl:
            # Format 1: Manual/notebook format with pre-built neighbor_list
            bonds_nl = nl["bonds"]
            self.bonds_i = bonds_nl["index_mapping"][0]
            self.bonds_j = bonds_nl["index_mapping"][1]
            
            angle_mapping = nl["angles"]["index_mapping"]
            self.angles_i = angle_mapping[0]
            self.angles_j = angle_mapping[1]
            self.angles_k = angle_mapping[2]
            
            self.dihedrals_mapping = nl["pseudo_ca_dihedral"]["index_mapping"]
        else:
            # Format 2: MLCG H5 format - build topology from exc_pairs
            if "exc_pairs" not in atomic_data:
                raise KeyError(
                    "MLCGGrappaWrapper requires either neighbor_list['bonds'] or 'exc_pairs' attribute. "
                    "Add 'exc_pairs: attrs:cg_exc_pairs' to your dataset.yaml loading_options."
                )
            
            exc_pairs = atomic_data["exc_pairs"]
            if isinstance(exc_pairs, np.ndarray):
                exc_pairs = torch.tensor(exc_pairs, dtype=torch.long)
            
            self.bonds_i = exc_pairs[0]
            self.bonds_j = exc_pairs[1]
            
            # Build angles from bonds
            bonds_i_np = self.bonds_i.cpu().numpy()
            bonds_j_np = self.bonds_j.cpu().numpy()
            
            bond_dict = {}
            for i, j in zip(bonds_i_np, bonds_j_np):
                bond_dict.setdefault(int(i), []).append(int(j))
                bond_dict.setdefault(int(j), []).append(int(i))
            
            angles = []
            for j in bond_dict:
                neighbors = bond_dict[j]
                for idx1, i in enumerate(neighbors):
                    for k in neighbors[idx1+1:]:
                        angles.append([i, j, k])
            angles = np.array(angles).T if angles else np.zeros((3, 0), dtype=int)
            
            self.angles_i = torch.tensor(angles[0], dtype=torch.long, device=atomic_data.pos.device)
            self.angles_j = torch.tensor(angles[1], dtype=torch.long, device=atomic_data.pos.device)
            self.angles_k = torch.tensor(angles[2], dtype=torch.long, device=atomic_data.pos.device)
            
            # Build dihedrals from bonds
            dihedrals = []
            for i, j in zip(bonds_i_np, bonds_j_np):
                for k in bond_dict.get(int(j), []):
                    if k != int(i):
                        for l in bond_dict.get(k, []):
                            if l != int(j):
                                dihedrals.append([int(i), int(j), k, l])
            dihedrals = np.array(dihedrals).T if dihedrals else np.zeros((4, 0), dtype=int)
            
            self.dihedrals_mapping = torch.tensor(dihedrals, dtype=torch.long, device=atomic_data.pos.device)
        
        # Build bonds list for Grappa Molecule
        bonds = [(int(i), int(j)) for i, j in zip(self.bonds_i.cpu().numpy(), self.bonds_j.cpu().numpy())]

        # Build Grappa Molecule (topology only)
        self.mol = Molecule(
            atoms=list(range(n_atoms)),
            bonds=bonds,
            impropers=[],
            partial_charges=[0.0] * n_atoms,
            atomic_numbers=atomic_data["atom_types"].cpu().numpy().tolist(),
        )

        # Convert to DGL graph (topology only, no positions yet)
        self.g = self.mol.to_dgl(
            max_element=self.grappa_model.max_element, exclude_feats=[]
        )

    def forward(self, atomic_data: AtomicData):
        """
        Forward pass computing total energy with gradient preservation.

        Parameters
        ----------
        atomic_data : AtomicData
            Must contain pos, atom_types, and neighbor_list

        Returns
        -------
        AtomicData
            Input data with predictions added to atomic_data.out[self.name]
        """

        # Build topology if not already done (first call)
        if self.g is None:
            self._build_topology(atomic_data)

        # Move graph to correct device
        device = atomic_data.pos.device
        g = self.g.to(device)

        # Get gradient-preserving parameters from raw model (same for all batch samples)
        params_with_grad = self.raw_model(g)

        # Extract positions and enable gradients for force computation
        pos = atomic_data.pos
        
        # Enable gradients on pos if not already enabled (needed for force computation)
        # Must do this BEFORE energy computation to preserve gradient graph
        if not pos.requires_grad:
            pos.requires_grad_(True)

        # ============================================================
        # HANDLE BATCHING: replicate topology indices for each batch sample
        # ============================================================
        total_atoms = pos.size(0)
        n_atoms = self._n_atoms_per_mol if self._n_atoms_per_mol else total_atoms
        batch_size = total_atoms // n_atoms
        
        if batch_size > 1:
            # Create offsets for each batch sample: [0, n_atoms, 2*n_atoms, ...]
            offsets = torch.arange(batch_size, device=device) * n_atoms
            
            # Replicate bond indices with offsets
            bonds_i = (self.bonds_i.to(device).unsqueeze(0) + offsets.unsqueeze(1)).flatten()
            bonds_j = (self.bonds_j.to(device).unsqueeze(0) + offsets.unsqueeze(1)).flatten()
            
            # Replicate angle indices
            angles_i = (self.angles_i.to(device).unsqueeze(0) + offsets.unsqueeze(1)).flatten()
            angles_j = (self.angles_j.to(device).unsqueeze(0) + offsets.unsqueeze(1)).flatten()
            angles_k = (self.angles_k.to(device).unsqueeze(0) + offsets.unsqueeze(1)).flatten()
            
            # Replicate dihedral indices: shape (4, n_dihedrals) -> (4, batch_size * n_dihedrals)
            dihedrals = (self.dihedrals_mapping.to(device).unsqueeze(0) + offsets.view(-1, 1, 1)).reshape(batch_size, 4, -1).permute(1, 0, 2).reshape(4, -1)
        else:
            bonds_i = self.bonds_i.to(device)
            bonds_j = self.bonds_j.to(device)
            angles_i = self.angles_i.to(device)
            angles_j = self.angles_j.to(device)
            angles_k = self.angles_k.to(device)
            dihedrals = self.dihedrals_mapping.to(device)

        # ============================================================
        # BONDS (with 0.5 factor correction)
        # ============================================================
        distances = torch.norm(pos[bonds_i] - pos[bonds_j], dim=1)
        bond_k = params_with_grad.nodes["n2"].data["k"]
        bond_eq = params_with_grad.nodes["n2"].data["eq"]
        # Tile parameters for batched data
        if batch_size > 1:
            bond_k = bond_k.repeat(batch_size)
            bond_eq = bond_eq.repeat(batch_size)
        # MLCG's Harmonic.compute omits 0.5, so we add it here
        bond_energies = 0.5 * Harmonic.compute(
            x=distances, x0=bond_eq, k=bond_k, V0=0.0
        )
        total_bond_energy = bond_energies.sum()

        # ============================================================
        # ANGLES (with 0.5 factor correction)
        # ============================================================
        vec_ji = pos[angles_i] - pos[angles_j]
        vec_jk = pos[angles_k] - pos[angles_j]
        cos_angles = torch.sum(vec_ji * vec_jk, dim=1) / (
            torch.norm(vec_ji, dim=1) * torch.norm(vec_jk, dim=1)
        )

        angle_k_param = params_with_grad.nodes["n3"].data["k"]
        angle_eq = params_with_grad.nodes["n3"].data["eq"]
        # Tile parameters for batched data
        if batch_size > 1:
            angle_k_param = angle_k_param.repeat(batch_size)
            angle_eq = angle_eq.repeat(batch_size)
        cos_angle_eq = torch.cos(angle_eq)
        # MLCG's HarmonicAnglesRaw.compute omits 0.5, so we add it here
        angle_energies = 0.5 * HarmonicAnglesRaw.compute(
            x=cos_angles, x0=cos_angle_eq, k=angle_k_param, V0=0.0
        )
        total_angle_energy = angle_energies.sum()

        # ============================================================
        # DIHEDRALS
        # ============================================================

        torsions = HarmonicImpropers.compute_features(pos, dihedrals)

        # Extract k values and derive phases from sign
        raw_proper_ks = params_with_grad.nodes["n4"].data["k"]
        raw_proper_phases = torch.where(
            raw_proper_ks >= 0.0,
            torch.zeros_like(raw_proper_ks),
            torch.full_like(raw_proper_ks, torch.pi),
        )
        raw_proper_ks = torch.abs(raw_proper_ks)

        # Tile parameters for batched data
        if batch_size > 1:
            raw_proper_ks = raw_proper_ks.repeat(batch_size, 1)
            raw_proper_phases = raw_proper_phases.repeat(batch_size, 1)

        # Convert to Fourier series format
        raw_k1s = raw_proper_ks * torch.sin(raw_proper_phases)
        raw_k2s = raw_proper_ks * torch.cos(raw_proper_phases)
        raw_v_0 = raw_proper_ks.sum(dim=1)

        dihedral_energies = FourierSeries.compute(
            theta=torsions, v_0=raw_v_0, k1s=raw_k1s, k2s=raw_k2s
        )
        total_dihedral_energy = dihedral_energies.sum()

        # ============================================================
        # TOTAL ENERGY
        # ============================================================
        total_energy = total_bond_energy + total_angle_energy + total_dihedral_energy

        # Store energy at top level for simulation (MLCG expects data.out['energy'])
        # Also store under model name for training compatibility with PLModel
        atomic_data.out['energy'] = total_energy.unsqueeze(0)
        
        # Always compute forces (needed for both training and inference)
        # Use retain_graph=True so other models (like SchNet) can also compute gradients
        # Use create_graph=True during training so we can backprop through forces
        forces = -torch.autograd.grad(
            total_energy,
            pos,
            create_graph=self.training,  # Allow backprop through forces during training
            retain_graph=True  # Keep graph for SchNet's gradient computation
        )[0]
        atomic_data.out['forces'] = forces
        atomic_data.out[self.name] = {"energy": total_energy.unsqueeze(0), "forces": forces}
        
        return atomic_data

class StandardGrappa(MLCGGrappaWrapper):
    """
    Customizable Grappa model that allows full control over architecture parameters.
    Inherits from MLCGGrappaWrapper to work seamlessly with MLCG's AtomicData format.
    
    Parameters
    ----------
    h5_file_path : str, optional
        Path to the HDF5 file containing topology (cg_exc_pairs). Required when
        using MLCG's data loader since it doesn't load exc_pairs.
    molecule_name : str, optional  
        Name of the molecule group in the HDF5 file. If None, will search for
        cg_exc_pairs in all groups.
    """
    
    def __init__(self,
                # ============================================================
                # TOPOLOGY SOURCE (for MLCG compatibility)
                # ============================================================
                 h5_file_path:str=None,
                 molecule_name:str=None,
                 
                # ============================================================
                # GNN ARCHITECTURE
                # ============================================================
                
                 graph_node_features:int=256,
                 in_feats:int=None,
                 in_feat_name:Union[str,List[str]]=["atomic_number", "ring_encoding", "partial_charge", "degree"],
                 in_feat_dims:Dict[str,int]={},
                 gnn_width:int=512,
                 gnn_attentional_layers:int=4,
                 gnn_convolutions:int=0,
                 gnn_attention_heads:int=16,

                # ============================================================
                # GNN REGULARIZATION
                # ============================================================
       
                 gnn_dropout_attention:float=0.3,
                 gnn_dropout_initial:float=0.,
                 gnn_dropout_conv:float=0.,
                 gnn_dropout_final:float=0.1,

                # ============================================================
                # PARAMETER WRITER ARCHITECTURE
                # ============================================================
       
                 symmetric_transformer_dropout:float=0.5,
                 symmetric_transformer_depth:int=1,
                 symmetric_transformer_n_heads:int=8,
                 symmetric_transformer_width:int=512,
                 symmetriser_depth:int=4,
                 symmetriser_width:int=256,


                  # ============================================================
                # TORSION PARAMETERS
                # ============================================================
       
                 n_periodicity_proper:int=3,
                 n_periodicity_improper:int=3,
                 gated_torsion:bool=False,

                 # ============================================================
                # OTHER FLAGS
                # ============================================================
                
                 positional_encoding:bool=True,
                 layer_norm:bool=True,
                 self_interaction:bool=True,
                 learnable_statistics:bool=False,
                 param_statistics:dict=None,
                 torsion_cutoff:float=1.e-4,
                 harmonic_gate:bool=False,
                 only_n2_improper:bool=True,
                 stat_scaling:bool=True,
                 shifted_elu:bool=True):
        
        # Initialize as torch.nn.Module (skip parent's __init__)
        torch.nn.Module.__init__(self)
        
        # Use default statistics if not provided
        if param_statistics is None:
            param_statistics = get_default_statistics()

        # Create custom GrappaModel with specified parameters
        self.raw_model = GrappaModel(
            graph_node_features=graph_node_features,
            in_feats=in_feats,
            in_feat_name=in_feat_name,
            in_feat_dims=in_feat_dims,
            gnn_width=gnn_width,
            gnn_attentional_layers=gnn_attentional_layers,
            gnn_convolutions=gnn_convolutions,
            gnn_attention_heads=gnn_attention_heads,
            gnn_dropout_attention=gnn_dropout_attention,
            gnn_dropout_initial=gnn_dropout_initial,
            gnn_dropout_conv=gnn_dropout_conv,
            gnn_dropout_final=gnn_dropout_final,
            symmetric_transformer_dropout=symmetric_transformer_dropout,
            symmetric_transformer_depth=symmetric_transformer_depth,
            symmetric_transformer_n_heads=symmetric_transformer_n_heads,
            symmetric_transformer_width=symmetric_transformer_width,
            symmetriser_depth=symmetriser_depth,
            symmetriser_width=symmetriser_width,
            n_periodicity_proper=n_periodicity_proper,
            n_periodicity_improper=n_periodicity_improper,
            gated_torsion=gated_torsion,
            positional_encoding=positional_encoding,
            layer_norm=layer_norm,
            self_interaction=self_interaction,
            learnable_statistics=learnable_statistics,
            param_statistics=param_statistics,
            torsion_cutoff=torsion_cutoff,
            harmonic_gate=harmonic_gate,
            only_n2_improper=only_n2_improper,
            stat_scaling=stat_scaling,
            shifted_elu=shifted_elu,
        )
        
        # Set up attributes expected by MLCGGrappaWrapper
        self.r_max = None  # Topology-based, no cutoff needed
        self.name = "grappa"  # PLModel uses this to find predictions
        self.derivative = True
        
        # Set max_element (from grappa constants, typically 100)
        self.max_element = 100
        
        # Store HDF5 path for loading topology
        self.h5_file_path = h5_file_path
        self.molecule_name = molecule_name
        self._exc_pairs_from_h5 = None  # Will be loaded on first forward pass
        self._n_atoms_per_mol = None  # Number of atoms in single molecule
        
        # Initialize topology attributes to None (will be built on first forward pass)
        self.g = None
        self.bonds_i = None
        self.bonds_j = None
        self.angles_i = None
        self.angles_j = None
        self.angles_k = None
        self.dihedrals_mapping = None
        
        # NOTE: We do NOT create self.grappa_model or self.model 
        # because we're replacing them with custom raw_model
    
    def _load_exc_pairs_from_h5(self):
        """Load cg_exc_pairs and n_atoms_per_mol directly from HDF5 file."""
        if self.h5_file_path is None:
            return None
        
        import h5py
        print("loading exc_pairs from h5 file:", self.h5_file_path)
        with h5py.File(self.h5_file_path, 'r') as f:
            if self.molecule_name:
                # Try specific molecule path
                for path in [f'{self.molecule_name}/{self.molecule_name}', self.molecule_name]:
                    if path in f and 'cg_exc_pairs' in f[path].attrs:
                        exc_pairs = f[path].attrs['cg_exc_pairs']
                        # Also get n_atoms from coords shape or embeds
                        if 'cg_coords' in f[path]:
                            self._n_atoms_per_mol = f[path]['cg_coords'].shape[1]
                        elif 'cg_embeds' in f[path].attrs:
                            self._n_atoms_per_mol = len(f[path].attrs['cg_embeds'])
                        return exc_pairs
            
            # Search all groups for cg_exc_pairs
            def find_exc_pairs_and_natoms(group):
                if 'cg_exc_pairs' in group.attrs:
                    exc_pairs = group.attrs['cg_exc_pairs']
                    # Also get n_atoms
                    if 'cg_coords' in group:
                        self._n_atoms_per_mol = group['cg_coords'].shape[1]
                    elif 'cg_embeds' in group.attrs:
                        self._n_atoms_per_mol = len(group.attrs['cg_embeds'])
                    return exc_pairs
                for key in group.keys():
                    if isinstance(group[key], h5py.Group):
                        result = find_exc_pairs_and_natoms(group[key])
                        if result is not None:
                            return result
                return None
            
            return find_exc_pairs_and_natoms(f)
    
    def _build_topology(self, atomic_data: AtomicData):
        """Build Grappa topology from first AtomicData sample (call once).
        
        Overrides parent method to use self.max_element instead of self.grappa_model.max_element
        since StandardGrappa doesn't create a grappa_model wrapper.
        
        Supports three data formats:
        1. Manual/notebook format: neighbor_list contains 'bonds', 'angles', 'pseudo_ca_dihedral' keys
        2. MLCG H5 format with exc_pairs in data: topology built from 'exc_pairs' attribute
        3. MLCG H5 format with h5_file_path: topology loaded directly from HDF5 file
        
        For batched data (MLCG), builds topology for a single molecule and stores
        n_atoms_per_mol for batch-aware forward pass.
        """
        total_atoms = atomic_data["atom_types"].size(0)
        exc_pairs = None
        
        # Check if neighbor_list has the expected structure (notebook format)
        nl = atomic_data.get("neighbor_list", {})
        if "bonds" in nl:
            # Format 1: Manual/notebook format with pre-built neighbor_list
            bonds_nl = nl["bonds"]
            self.bonds_i = bonds_nl["index_mapping"][0]
            self.bonds_j = bonds_nl["index_mapping"][1]
            
            angle_mapping = nl["angles"]["index_mapping"]
            self.angles_i = angle_mapping[0]
            self.angles_j = angle_mapping[1]
            self.angles_k = angle_mapping[2]
            
            self.dihedrals_mapping = nl["pseudo_ca_dihedral"]["index_mapping"]
            
            # For notebook format, use total atoms (no batching)
            n_atoms = total_atoms
            atom_types = atomic_data["atom_types"].cpu().numpy().tolist()
        else:
            # Format 2/3: Build topology from exc_pairs
            # Try to get exc_pairs from: data attribute, HDF5 file, or cached value
            if "exc_pairs" in atomic_data:
                exc_pairs = atomic_data["exc_pairs"]
            elif self._exc_pairs_from_h5 is not None:
                exc_pairs = self._exc_pairs_from_h5
            elif self.h5_file_path is not None:
                exc_pairs = self._load_exc_pairs_from_h5()
                self._exc_pairs_from_h5 = exc_pairs  # Cache for future calls
            
            if exc_pairs is None:
                raise KeyError(
                    "StandardGrappa requires topology information. Provide one of:\n"
                    "  1. neighbor_list with 'bonds', 'angles', 'pseudo_ca_dihedral' keys\n"
                    "  2. h5_file_path parameter pointing to HDF5 with cg_exc_pairs\n"
                    "  3. 'exc_pairs' key in AtomicData"
                )
            
            if isinstance(exc_pairs, np.ndarray):
                exc_pairs = torch.tensor(exc_pairs, dtype=torch.long)
            
            self.bonds_i = exc_pairs[0]
            self.bonds_j = exc_pairs[1]
            
            # Determine n_atoms_per_mol for batched data
            if self._n_atoms_per_mol is None:
                # Infer from exc_pairs - atoms should be 0 to max(exc_pairs)
                self._n_atoms_per_mol = int(exc_pairs.max().item()) + 1
            n_atoms = self._n_atoms_per_mol
            
            # Get atom types for single molecule (first n_atoms of batch)
            atom_types = atomic_data["atom_types"][:n_atoms].cpu().numpy().tolist()
            
            # Build angles from bonds (i-j-k where j is central atom)
            bonds_i_np = self.bonds_i.cpu().numpy()
            bonds_j_np = self.bonds_j.cpu().numpy()
            
            bond_dict = {}
            for i, j in zip(bonds_i_np, bonds_j_np):
                bond_dict.setdefault(int(i), []).append(int(j))
                bond_dict.setdefault(int(j), []).append(int(i))
            
            angles = []
            for j in bond_dict:
                neighbors = bond_dict[j]
                for idx1, i in enumerate(neighbors):
                    for k in neighbors[idx1+1:]:
                        angles.append([i, j, k])
            angles = np.array(angles).T if angles else np.zeros((3, 0), dtype=int)
            
            self.angles_i = torch.tensor(angles[0], dtype=torch.long, device=atomic_data.pos.device)
            self.angles_j = torch.tensor(angles[1], dtype=torch.long, device=atomic_data.pos.device)
            self.angles_k = torch.tensor(angles[2], dtype=torch.long, device=atomic_data.pos.device)
            
            # Build dihedrals from bonds (i-j-k-l)
            dihedrals = []
            for i, j in zip(bonds_i_np, bonds_j_np):
                for k in bond_dict.get(int(j), []):
                    if k != int(i):
                        for l in bond_dict.get(k, []):
                            if l != int(j):
                                dihedrals.append([int(i), int(j), k, l])
            dihedrals = np.array(dihedrals).T if dihedrals else np.zeros((4, 0), dtype=int)
            
            self.dihedrals_mapping = torch.tensor(dihedrals, dtype=torch.long, device=atomic_data.pos.device)
        
        print(f"Building Grappa topology: {n_atoms} atoms, {len(self.bonds_i)} bonds")
        
        # Build bonds list for Grappa Molecule
        bonds = [(int(i), int(j)) for i, j in zip(self.bonds_i.cpu().numpy(), self.bonds_j.cpu().numpy())]

        # Build Grappa Molecule (topology only) - for SINGLE molecule
        self.mol = Molecule(
            atoms=list(range(n_atoms)),
            bonds=bonds,
            impropers=[],
            partial_charges=[0.0] * n_atoms,
            atomic_numbers=atom_types,
        )

        # Convert to DGL graph - use self.max_element instead of self.grappa_model.max_element
        self.g = self.mol.to_dgl(
            max_element=self.max_element,  # ← Uses StandardGrappa's max_element
            exclude_feats=[]
        )


class SumOutWrapper(torch.nn.Module):
    def __init__(self, models: Dict[str, torch.nn.Module], targets: List[str] = None):
        super().__init__()
        from mlcg.nn.gradients import SumOut

        self.model = SumOut(models=torch.nn.ModuleDict(models), targets=targets)
        self.name = "SumOut"
        self.targets = targets if targets else ['forces']

    def forward(self, data: AtomicData) -> AtomicData:
        data = self.model(data)
        # PLModel expects data.out['SumOut'] = {'forces': ..., 'energy': ...}
        # But MLCG's SumOut stores directly in data.out['forces'] etc.
        # So we need to create the 'SumOut' key
        data.out[self.name] = {target: data.out[target] for target in self.targets if target in data.out}
        return data

    def neighbor_list(self, **kwargs):
        return self.model.neighbor_list(**kwargs)