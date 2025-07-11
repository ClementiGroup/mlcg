import numpy as np
import pickle as pkl
import torch
import os
import mdtraj as md
from mlcg.data import AtomicData
from torch_geometric.data.collate import collate
from allegro.model import AllegroModel
from nequip.data import AtomicDataDict
from hydra.utils import instantiate
from allegro.model.allegro_models import FullAllegroModel, AllegroModel
from tqdm import tqdm
from torch_cluster import radius_graph
from nequip.utils.global_state import set_global_state, get_latest_global_state, global_state_initialized
import h5py
import pytorch_lightning as pl
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

ELEMENT_MAP = {
    1: "H",   2: "He",  3: "Li",  4: "Be",  5: "B",   6: "C",   7: "N",   8: "O",   9: "F",   10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",  16: "S",  17: "Cl", 18: "Ar", 19: "K",  20: "Ca",
    21: "Sc", 22: "Ti", 23: "V",  24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
    31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y",  40: "Zr",
    41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I",  54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd",
    61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb",
    71: "Lu", 72: "Hf", 73: "Ta", 74: "W",  75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg",
    81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th",
    91: "Pa", 92: "U",  93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf", 99: "Es", 100: "Fm",
    101: "Md", 102: "No", 103: "Lr", 104: "Rf", 105: "Db", 106: "Sg", 107: "Bh", 108: "Hs", 109: "Mt", 110: "Ds",
    111: "Rg", 112: "Cn", 113: "Nh", 114: "Fl", 115: "Mc", 116: "Lv", 117: "Ts", 118: "Og"
}



class StandardAllegro(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        type_names: list[str] = None,
        r_max: float = 4.0,
        avg_num_neighbors: float = 20.0,
        num_layers: int = 2,
        l_max: int = 2,
        num_scalar_features: int = 32,
        num_tensor_features: int = 4,
        radial_chemical_embed_dim: int = 16,
        scalar_embed_mlp_hidden_layers_depth: int = 1,
        scalar_embed_mlp_hidden_layers_width: int = 32,
        allegro_mlp_hidden_layers_depth: int = 2,
        allegro_mlp_hidden_layers_width: int = 32,
        readout_mlp_hidden_layers_depth: int = 1,
        readout_mlp_hidden_layers_width: int = 8,
        model_dtype: str = "float32",
        seed: int = 123,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()  # Important for Lightning

        set_global_state()
        
        # Extract learning rate and other training params
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.weight_decay = kwargs.pop('weight_decay', 0.0)

        default_params = {
            "seed": 123,
            "type_names": ["H", "C", "N", "O"],
            "model_dtype": "float32",
            "r_max": 4.0,
            "avg_num_neighbors": 20.0,
            "radial_chemical_embed_dim": 16,
            "scalar_embed_mlp_hidden_layers_depth": 1,
            "scalar_embed_mlp_hidden_layers_width": 32,
            "num_layers": 2,
            "l_max": 2,
            "num_scalar_features": 32,
            "num_tensor_features": 4,
            "allegro_mlp_hidden_layers_depth": 2,
            "allegro_mlp_hidden_layers_width": 32,
            "readout_mlp_hidden_layers_depth": 1,
            "readout_mlp_hidden_layers_width": 8,
            "radial_chemical_embed": {
                "_target_": "allegro.nn.TwoBodyBesselScalarEmbed",
                "num_bessels": 8,
            },
        }

        self.atom_type_mapping = None


        # Override default parameters with any provided kwargs
        default_params.update(kwargs)

        self.r_max = default_params["r_max"]
        self.allegro_model = AllegroModel(**default_params)

                # ADD THIS LINE

    def map_atom_types(self, raw_atom_types):
        if self.atom_type_mapping is None:
            unique_types = torch.unique(raw_atom_types).cpu().numpy()
            self.atom_type_mapping = {int(atomic_num): i for i, atomic_num in enumerate(sorted(unique_types))}
            # If type_names is not set, generate it from atomic numbers
            if not getattr(self, "hparams", None) or not self.hparams.get("type_names"):
                self.hparams.type_names = [ELEMENT_MAP.get(int(num), f"X{num}") for num in sorted(unique_types)]
                print(f"[Auto] Generated type_names: {self.hparams.type_names}")
            print(f"Auto-created mapping: {self.atom_type_mapping}")
        mapped = torch.zeros_like(raw_atom_types)
        for i, raw_type in enumerate(raw_atom_types):
            mapped[i] = self.atom_type_mapping[raw_type.item()]
        return mapped

    # # does the class need a forward method to be defined? I could dispatch to the correct method based on the input type
    # def forward(self, atomic_data):
    #     return self.forward_collated(atomic_data)

    def forward(self, atomic_data: AtomicData, is_collated: bool = True):
        return self.forward_collated(atomic_data)
        # else:
        #     return self.forward_single_frame(atomic_data)

    def forward_single_frame(self, atomic_data: AtomicData):
        # Convert AtomicData to dict format
       # Extract required inputs from atomic_data
        positions = atomic_data[AtomicDataDict.POSITIONS_KEY]        # shape: (n_atoms, 3)
        mapped_types = atomic_data[AtomicDataDict.ATOM_TYPE_KEY]     # already mapped to Allegro type indices

        # raw_atom_types = atomic_data[AtomicDataDict.ATOM_TYPE_KEY]
        # mapped_types = self.map_atom_types(raw_atom_types)

        # Create batch: assume all atoms belong to the same frame
        batch = torch.zeros(positions.shape[0], dtype=torch.long, device=positions.device)

        # Compute edges using radius graph
        edge_index = radius_graph(positions, r=4.0, batch=batch, loop=False)

        # Pack input dict for Allegro
        model_input = {
            AtomicDataDict.POSITIONS_KEY: positions,
            AtomicDataDict.EDGE_INDEX_KEY: edge_index,
            AtomicDataDict.ATOM_TYPE_KEY: mapped_types,
        }

        # Run model
        output = self.allegro_model(model_input)
        return output
    
    
    def forward_collated(self, atomic_data: AtomicData, num_frames_to_use=2):

        positions_flat = atomic_data[AtomicDataDict.POSITIONS_KEY]       # shape: (F*A, 3)
        batch_flat = atomic_data[AtomicDataDict.BATCH_KEY]               # shape: (F*A,)
        mapped_types = atomic_data[AtomicDataDict.ATOM_TYPE_KEY]

        raw_atom_types = atomic_data[AtomicDataDict.ATOM_TYPE_KEY]
        mapped_types = self.map_atom_types(raw_atom_types)

        print("raw_atom_types[:20]:", raw_atom_types[:20])
        print("mapped_types[:20]:", mapped_types[:20])
        print("mapped_types min/max:", mapped_types.min().item(), mapped_types.max().item())
        print("embedding table size:", len(self.hparams.type_names))

        print("raw_atom_types[:20]:", raw_atom_types[:20])
        print("type_names:", self.hparams.type_names)
        print("atom_type_mapping:", self.atom_type_mapping)

        #   # DEBUG: Check actual data sizes
        # print(f"Total positions: {positions_flat.shape[0]}")
        # print(f"Requesting {num_frames_to_use} frames")
        # print(f"Batch range: {batch_flat.min().item()} to {batch_flat.max().item()}")
        

        # num_frames_to_use <= 2
        mask = batch_flat < num_frames_to_use  # (F*A,) boolean mask

        print(f"Mask selects {mask.sum().item()} atoms out of {mask.shape[0]} total")


        positions_subset = positions_flat[mask]
        # forces_subset = forces_flat[mask]
        batch_subset = batch_flat[mask]
        mapped_types_subset = mapped_types[mask]

        num_nodes_per_frame = torch.tensor([(batch_subset == i).sum().item() for i in range(num_frames_to_use)])


        # _, batch_subset_reindexed = torch.unique(batch_subset, return_inverse=True)
        edge_index_subset = radius_graph(
            positions_subset,
            r=self.r_max,
            batch=batch_subset
        )

        center_types = mapped_types_subset[edge_index_subset[0]]
        neighbor_types = mapped_types_subset[edge_index_subset[1]]
        print("center_types min/max:", center_types.min().item(), center_types.max().item())
        print("neighbor_types min/max:", neighbor_types.min().item(), neighbor_types.max().item())
        print("Any negative center_types?", (center_types < 0).any().item())
        print("Any negative neighbor_types?", (neighbor_types < 0).any().item())
        print("Any center_types >= embedding size?", (center_types >= len(self.hparams.type_names)).any().item())
        print("Any neighbor_types >= embedding size?", (neighbor_types >= len(self.hparams.type_names)).any().item())

        allegro_subset_data = {
            AtomicDataDict.POSITIONS_KEY: positions_subset,
            AtomicDataDict.ATOM_TYPE_KEY: mapped_types_subset,
            AtomicDataDict.EDGE_INDEX_KEY: edge_index_subset,
            AtomicDataDict.BATCH_KEY: batch_subset,
            AtomicDataDict.NUM_NODES_KEY: num_nodes_per_frame,
        }

        print("\n\n\n", allegro_subset_data, "\n\n\n")

        collated_output = self.allegro_model(allegro_subset_data)

        return collated_output
    
    def training_step(self, batch, batch_idx):
        """Training step for Lightning"""
        output = self.forward(batch)
        
        # Force loss
        if 'forces' in batch and 'forces' in output:
            force_loss = torch.nn.functional.mse_loss(output['forces'], batch['forces'])
            self.log('train_force_loss', force_loss, prog_bar=True)
        else:
            force_loss = 0.0
            
        # Energy loss (if available)
        if 'total_energy' in batch and 'total_energy' in output:
            energy_loss = torch.nn.functional.mse_loss(output['total_energy'], batch['total_energy'])
            self.log('train_energy_loss', energy_loss, prog_bar=True)
        else:
            energy_loss = 0.0
            
        total_loss = force_loss + energy_loss
        self.log('train_loss', total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step for Lightning"""
        output = self.forward(batch)
        
        if 'forces' in batch and 'forces' in output:
            val_force_loss = torch.nn.functional.mse_loss(output['forces'], batch['forces'])
            self.log('val_force_loss', val_force_loss, sync_dist=True)
            
        if 'total_energy' in batch and 'total_energy' in output:
            val_energy_loss = torch.nn.functional.mse_loss(output['total_energy'], batch['total_energy'])
            self.log('val_energy_loss', val_energy_loss, sync_dist=True)

    def configure_optimizers(self):
        """Configure optimizer for Lightning"""
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_force_loss"
        }

    
def small_test():
    set_global_state()

    CG_COORDS_PATH = "/srv/data/kamenrur95/atom_encoding_example/1FME_cg_coords.npy"
    CG_EMBEDS_PATH = "/srv/data/kamenrur95/atom_encoding_example/1FME_cg_embeds.npy"
    PRIOS_PATH = "/srv/data/kamenrur95/atom_encoding_example/1FME_prior_nls_badn_min_pair_4.pkl"

    cg_coords = np.load(CG_COORDS_PATH)        # shape: (n_frames, n_atoms, 3)
    cg_embeds = np.load(CG_EMBEDS_PATH)        # shape: (n_atoms,)
    with open(PRIOS_PATH, "rb") as f:
        nls = pkl.load(f)
    
    print("cg_embads: ", cg_embeds)
    print(f"Loaded data: {cg_coords.shape[0]} frames, {cg_coords.shape[1]} atoms")

    # print("Unique atomic numbers found:", unique_atoms)


# --- Mapping logic ---
    unique_atomic_numbers = sorted(set(int(z) for z in cg_embeds))
    print("Unique atomic numbers found:", unique_atomic_numbers)

    type_names = [ELEMENT_MAP.get(num, f"X{num}") for num in unique_atomic_numbers]
    print("Type names:", type_names)

    mapping = {num: idx for idx, num in enumerate(unique_atomic_numbers)}
    print("Mapping:", mapping)

    mapped_embeds = np.array([mapping[int(z)] for z in cg_embeds])
    # --- End mapping logic ---

    # Create AtomicData
    frame_idx = 0
    single_frame_atom_data = AtomicData.from_points(
        pos=cg_coords[frame_idx].astype(np.float32),
        atom_types=mapped_embeds,
        neighborlist=nls
    )
        
    num_frames_available = cg_coords.shape[0]
    print(f"Creating collated data with all {num_frames_available} frames...")
    
    frame_data_list = []
    for frame_idx in range(num_frames_available):
        frame_data = AtomicData.from_points(
            pos=cg_coords[frame_idx].astype(np.float32),
            atom_types=mapped_embeds,
            neighborlist=nls
        )
        frame_data_list.append(frame_data)
    
    # Collate ALL frames
    collated_data, _, _ = collate(AtomicData, frame_data_list)

    model = StandardAllegro(type_names=type_names)

    print("Running single frame inference...")
    # Single frame inference
    single_output = model.forward_single_frame(single_frame_atom_data)
    print(single_output)
    
    print("Running collated inference (2 frames)...")
    # Collated inference with subset of frames
    collated_output = model.forward_collated(collated_data, num_frames_to_use=2)
    print(f"Collated energy: {collated_output.get('total_energy', 'N/A')}")
    
    print("Running collated inference (100 frames)...")
    # Collated inference with subset of frames
    collated_output = model.forward_collated(collated_data, num_frames_to_use=100)
    # print(f"Collated energy: {collated_output.get('total_energy', 'N/A')}")

def load_bba_dataset():
    """Load BBA dataset exactly like the notebook"""
    raw_data_dir = "/group/ag_clementi_cmb/projects/single_protein_datasets/charmm/raw_charmm22star_bba"
    pdb_template_fn = f"{raw_data_dir}/bba_50ns_0/structure.pdb"
    coords_dir = os.path.join(raw_data_dir, "coords_nowater")

    pdb = md.load(pdb_template_fn)
    non_water_idxs = [atom.index for atom in pdb.topology.atoms if atom.residue.name not in ("HOH", "CLA")]
    pdb_nowater = pdb.atom_slice(non_water_idxs)
    atomic_numbers = np.array([atom.element.atomic_number for atom in pdb_nowater.topology.atoms])

    coord_files = sorted([f for f in os.listdir(coords_dir) if f.endswith('.npy')])[:100]
    
    all_positions = []
    for coord_file in tqdm(coord_files):
        coords = np.load(os.path.join(coords_dir, coord_file))
        all_positions.append(coords)

    positions = np.concatenate(all_positions, axis=0)
    print(f"Loaded BBA: {positions.shape[0]} frames, {positions.shape[1]} atoms")
    
    return positions, atomic_numbers

def create_bba_atomic_data(positions, atomic_numbers):
    """Create AtomicData like the notebook"""
    num_frames, num_atoms = positions.shape[0], positions.shape[1]
    
    # Flatten everything like the notebook
    positions_flat = torch.tensor(positions.reshape(-1, 3), dtype=torch.float32)
    batch_flat = torch.repeat_interleave(torch.arange(num_frames), num_atoms)
    
    # Map atomic numbers
    mapping = {1:0, 6:1, 7:2, 8:3}  # H, C, N, O from notebook
    mapped_types = torch.tensor([mapping[int(z)] for z in atomic_numbers], dtype=torch.long)
    mapped_types_flat = mapped_types.repeat(num_frames)
    
    # Create simple AtomicData-like object
    class SimpleAtomicData:
        def __init__(self, pos, batch, atom_types):
            self.data = {
                AtomicDataDict.POSITIONS_KEY: pos,
                AtomicDataDict.BATCH_KEY: batch,
                AtomicDataDict.ATOM_TYPE_KEY: atom_types
            }
        def __getitem__(self, key):
            return self.data[key]
    
    return SimpleAtomicData(positions_flat, batch_flat, mapped_types_flat)

def bigger_test():
    set_global_state()
    
    print("=== LOADING BBA DATASET ===")
    positions, atomic_numbers = load_bba_dataset()
    
    print("=== CREATING BBA ATOMIC DATA ===")
    bba_atomic_data = create_bba_atomic_data(positions, atomic_numbers)
    
    print("=== TESTING BBA WITH STANDARD ALLEGRO ===")
    model = StandardAllegro(type_names=["H", "C", "N", "O"])
    
    # Test with different frame counts (like the notebook)
    test_frames = [1, 2, 5, 10, 50, 100, 150]
    
    for num_frames in test_frames:
        print(f"\nTesting BBA with {num_frames} frames:")
        import time
        start = time.time()
        
        output = model.forward_collated(bba_atomic_data, num_frames_to_use=num_frames)
        
        elapsed = time.time() - start
        print(f"  BBA inference ({num_frames} frames): {elapsed:.3f} seconds")
        
        if 'total_energy' in output:
            energy_shape = output['total_energy'].shape
            print(f"  Energy shape: {energy_shape}")
            if energy_shape[0] > 0:
                print(f"  Sample energies: {output['total_energy'][:3]}")

class AllegroDataset(Dataset):
    def __init__(self, coords, forces):
        self.coords = torch.tensor(coords, dtype=torch.float32)
        self.forces = torch.tensor(forces, dtype=torch.float32)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return {
            "coords": self.coords[idx],
            "forces": self.forces[idx]
        }

def collate_fn(batch):
    # batch is a list of AtomicData objects
    # Collate to a single AtomicDataDict
    return AtomicData.from_AtomicData_list(batch).to_AtomicDataDict()

# train_loader = DataLoader(
#     AllegroDataset(train_data_list),
#     batch_size=4,
#     shuffle=True,
#     collate_fn=collate_fn,
# )

def collate_fn(batch):
    # batch is a list of AtomicData objects
    # Collate to a single AtomicDataDict
    return AtomicData.from_AtomicData_list(batch).to_AtomicDataDict()

mlcg_single_molecule_example_h5_file_path = "/srv/data/kamenrur95/mlcg/examples/h5_pl/single_molecule/1L2Y_prior_tag.h5"

def main():
    small_test()
    # bigger_test()
    with h5py.File(mlcg_single_molecule_example_h5_file_path, "r") as f:
        coords = np.array(f["1L2Y/1L2Y/cg_coords"])
        forces = np.array(f["1L2Y/1L2Y/cg_delta_forces"])
    
    print("Coordinates shape:", coords.shape)
    print("Forces shape:", forces.shape)
    dataset = AllegroDataset(coords, forces)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



if __name__ == "__main__":
    main()