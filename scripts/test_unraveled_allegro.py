import torch

import sys
sys.path.insert(0, "/local_scratch2/jacopo/software/mlcg")

from mlcg.nn import ForceMSE
from mlcg.nn.gradients import GradientsOut
from mlcg.data._keys import FORCE_KEY

from mlcg.nn.allegro import StandardAllegro as RaveledModel
from mlcg.nn.allegro import StandardAllegro as UnraveledModel

from mlcg.nn import CosineCutoff, ExpNormalBasis

from rangemp.datasets import AQMgasDataset
from mlcg.pl.data import DataModule
from mlcg.nn import Loss
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# SET PATHS FOR YOUR OWN ENVIRONMENT!!!
DATASET_PATH = "/srv/data/kamenrur95/datasets/AQMgas/mlcg/"
DATASET_SPLITS = "/srv/data/kamenrur95/datasets/AQMgas/mlcg/selection_n_atoms_2_train_0.85_val_0.1_seed_4272389.npz"

dataset = AQMgasDataset(DATASET_PATH)

loader = DataModule(
                dataset,
                splits=DATASET_SPLITS,
                loading_stride=1,
                batch_size=64,
                inference_batch_size=64,
                num_workers=6,
                save_local_copy="False",
                pin_memory="True",
                log_dir='.'
                )
loader.load_dataset()
loader.setup(stage="test")  # 499 strutture


# Add this function at the top level of your script (after imports)
def debug_energy_and_gradients(data, model_name):
    """Check if energy exists and is differentiable in output data"""
    print(f"\n=== ENERGY DEBUG FOR {model_name} ===")
    
    # Check if the model's output exists
    if model_name not in data.out:
        print(f"❌ No '{model_name}' key in data.out!")
        print(f"Available keys: {list(data.out.keys())}")
        return
    
    # Check if energy exists in the output
    if 'energy' not in data.out[model_name]:
        print(f"❌ No 'energy' key in data.out[{model_name}]!")
        print(f"Available keys: {list(data.out[model_name].keys())}")
        return
    
    # Get energy tensor
    energy = data.out[model_name]['energy']
    print(f"✓ Found energy tensor with shape: {energy.shape}")
    print(f"✓ Energy mean: {energy.mean().item():.4f}, std: {energy.std().item():.4f}")
    print(f"✓ Energy requires_grad: {energy.requires_grad}")
    
    # Check if forces are computed
    if 'forces' in data.out[model_name]:
        forces = data.out[model_name]['forces']
        print(f"✓ Found forces tensor with shape: {forces.shape}")
        print(f"✓ Forces mean: {forces.mean().item():.4f}, std: {forces.std().item():.4f}")
        print(f"✓ Forces requires_grad: {forces.requires_grad}")
    else:
        print("❌ No 'forces' key found!")
    
    # Test if we can compute gradients from energy
    if energy.requires_grad and data.pos.requires_grad:
        try:
            energy_sum = energy.sum()
            print(f"✓ Computing gradients from energy_sum: {energy_sum.item():.4f}")
            
            # Try to compute gradients
            grads = torch.autograd.grad(
                energy_sum, data.pos, create_graph=True, retain_graph=True
            )[0]
            
            print(f"✓ Successfully computed gradients from energy!")
            print(f"✓ Gradients shape: {grads.shape}")
            print(f"✓ Gradients mean: {grads.mean().item():.4f}, std: {grads.std().item():.4f}")
        except Exception as e:
            print(f"❌ Failed to compute gradients: {e}")
    else:
        if not energy.requires_grad:
            print("❌ Energy doesn't require gradients!")
        if not data.pos.requires_grad:
            print("❌ Positions don't require gradients!")

def test_both_models(hparam_dict, loader, device='cpu'):
    # Create both models
    raveled_model = RaveledModel(**hparam_dict).to(device)
    unraveled_model = GradientsOut(UnraveledModel(**hparam_dict), targets=FORCE_KEY).to(device)
    
    # Instead of using the full dataloader batch, create a smaller one
    small_batch_size = 8  # Much smaller than your current 512
    
    # Get one small batch
    for data in loader.test_dataloader():
        # Check the data type and handle appropriately
        print(f"Data type: {type(data)}")
        
        # If data is a list (which appears to be the case), take just one item
        if isinstance(data, list):
            batch = data[0].to(device)
        # If it's a Batch object with a slice method
        elif hasattr(data, '__getitem__') and hasattr(data, 'to'):
            try:
                # Try to take a slice if possible
                batch = data[0:1].to(device)
            except:
                # If slicing doesn't work, just take the whole batch
                batch = data.to(device)
        else:
            # Fallback, just use the whole batch
            batch = data
            if hasattr(batch, 'to'):
                batch = batch.to(device)
        break
    
    # Print batch information for debugging
    print(f"Batch type: {type(batch)}")
    if hasattr(batch, 'pos'):
        print(f"Batch positions shape: {batch.pos.shape}")
        
    # Set requires_grad for positions
    print("Setting requires_grad for positions...")
    if hasattr(batch, 'pos'):
        batch.pos.requires_grad_(True)
        print(f"Position requires_grad: {batch.pos.requires_grad}")
    
    # Test raveled model
    print("\n=== TESTING RAVELED MODEL ===")
    if hasattr(batch, 'clone'):
        raveled_batch = batch.clone()
    else:
        # If clone isn't available, create a copy manually
        import copy
        raveled_batch = copy.deepcopy(batch)
    
    if hasattr(raveled_batch, 'pos'):
        # Store the original position tensor for gradient tracking
        original_pos_raveled = raveled_batch.pos.clone().detach().requires_grad_(True)
        raveled_batch.pos = original_pos_raveled
        print(f"Position requires_grad before forward: {raveled_batch.pos.requires_grad}")
    
    # Run the model
    raveled_out = raveled_model(raveled_batch)
    
    # Verify position requires_grad after forward pass
    if hasattr(raveled_batch, 'pos'):
        print(f"Position requires_grad after forward: {raveled_batch.pos.requires_grad}")
    
    # Debug energy and gradients, passing the original position tensor
    debug_energy_and_gradients(raveled_out, raveled_model.name)
    
    # Clear cache between tests
    torch.cuda.empty_cache()
    
    # Test unraveled model
    print("\n=== TESTING UNRAVELED MODEL ===")
    if hasattr(batch, 'clone'):
        unraveled_batch = batch.clone()
    else:
        # If clone isn't available, create a copy manually
        import copy
        unraveled_batch = copy.deepcopy(batch)
    
    if hasattr(unraveled_batch, 'pos'):
        # Store the original position tensor for gradient tracking
        original_pos_unraveled = unraveled_batch.pos.clone().detach().requires_grad_(True)
        unraveled_batch.pos = original_pos_unraveled
        print(f"Position requires_grad before forward: {unraveled_batch.pos.requires_grad}")
    
    # Run the model
    unraveled_out = unraveled_model(unraveled_batch)
    
    # Verify position requires_grad after forward pass
    if hasattr(unraveled_batch, 'pos'):
        print(f"Position requires_grad after forward: {unraveled_batch.pos.requires_grad}")
    
    # Debug energy and gradients, passing the original position tensor
    debug_energy_and_gradients(unraveled_out, unraveled_model.name)
    
    return raveled_model, unraveled_model

def test_training(
                hparam_dict,
                loader,
                device='cpu'
                ):

    

    raveled_model = RaveledModel(**hparam_dict)
    unraveled_model = GradientsOut(UnraveledModel(**hparam_dict), targets=FORCE_KEY)
    model = unraveled_model
    print(model)
    # model = EnergyOut(test_model, targets=['energy'])
    loss_fn = Loss([
        ForceMSE(force_kwd='forces'),
        ForceMSE(force_kwd='energy'),
    ],
    weights=[
        1.0, 0.01
    ])


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Test one epoch training
    # print("____________________")
    model.train()
    model.to(device)

    start = time.time()
    pbar = tqdm(range(200))
    for epoch in pbar:
        losses = []
        for data in loader.test_dataloader():
            data.to(device)
            data.pos.requires_grad_(True)
            optimizer.zero_grad()
            data = model(data)

            # debug_energy_and_gradients(data, model.name)
            # first_batch = False

            data.out.update(**data.out[model.name])
            loss = loss_fn(data)
            losses.append(loss)
            loss.backward()
            optimizer.step()
        pbar.set_description(f"loss: {torch.tensor(losses).mean():.7f}")
        # print(model.model.allegro.latents[0].mlp[0].weight)
        # print(torch.tensor(losses).mean())
    end = time.time()
    
    print(f"Train time 200 epochs {end-start}")
    print("-------------------------------------------")

    # torch.save(data.pos, "positions.dat")
    # torch.save(data.atom_types, "atom_typ.dat")
    # torch.save(data.n_atoms, "n_atoms.dat")

def compare_model_dimensions(raveled_model, unraveled_model, loader, device='cpu'):
    """Show just the dimensional differences between the models"""
    # Get a batch
    for data in loader.test_dataloader():
        if isinstance(data, list):
            batch = data[0].to(device)
        else:
            batch = data.to(device)
        break
    
    # Enable gradient tracking
    batch.pos.requires_grad_(True)
    
    # Run both models
    raveled_out = raveled_model(batch.clone())
    unraveled_out = unraveled_model(batch.clone())
    
    # Get basic batch info
    num_atoms = batch.pos.shape[0]
    num_molecules = batch.batch.max().item() + 1
    
    # Print only the shapes
    print("\n=== MODEL OUTPUT DIMENSIONS ===")
    print(f"Batch contains {num_molecules} molecules with {num_atoms} atoms")
    print(f"\nRaveled model:")
    r_energy = raveled_out.out[raveled_model.name]['energy']
    r_forces = raveled_out.out[raveled_model.name]['forces']
    print(f"- Energy shape: {r_energy.shape}")
    print(f"- Forces shape: {r_forces.shape}")
    
    print(f"\nUnraveled model:")
    u_energy = unraveled_out.out[unraveled_model.name]['energy']
    u_forces = unraveled_out.out[unraveled_model.name]['forces']
    print(f"- Energy shape: {u_energy.shape}")
    print(f"- Forces shape: {u_forces.shape}")
    

if __name__ == "__main__":
                
    ## Allegro
    hparam_dict = {
        "r_max": 4.0,
        "embedding_size": 20,
        "l_max": 2,
        "parity": False,
        "scalar_embed_mlp_hidden_layers_depth": 1,
        "scalar_embed_mlp_hidden_layers_width": 32,
        "num_layers": 2,
        "num_scalar_features": 32,
        "num_tensor_features": 4,
        "allegro_mlp_hidden_layers_depth": 2,
        "allegro_mlp_hidden_layers_width": 32,
        "readout_mlp_hidden_layers_depth": 1,
        "readout_mlp_hidden_layers_width": 8,
        "avg_num_neighbors": 10.0,
        # "weight_individual_irreps": True,
    }

    ## So3
    # hparam_dict = {
    #     "rbf_layer": ExpNormalBasis(4.0, 33),
    #     "cutoff": CosineCutoff(0, 4.0),
    #     "output_hidden_layer_widths": [128, 64],
    #     "hidden_channels": 132,
    #     "embedding_size": 20,
    #     "num_interactions": 1,
    #     "degrees": [1, 2, 3],
    #     "n_heads": 4,
    #     "activation": torch.nn.SiLU(),
    #     "max_num_neighbors": 1000,
    #     "normalize_sph": True,
    # }

    ## MACE
    # hparam_dict = {
    #     "r_max": 4,
    #     "num_bessel": 33,
    #     "num_polynomial_cutoff": 5,
    #     "max_ell": 2,
    #     "interaction_cls": "mace.modules.blocks.RealAgnosticResidualInteractionBlock",
    #     "interaction_cls_first": "mace.modules.blocks.RealAgnosticResidualInteractionBlock",
    #     "num_interactions": 1,
    #     "hidden_irreps": "128x0e + 128x1o + 128x2e",
    #     "MLP_irreps": "64x0e",
    #     "avg_num_neighbors": 10.0,
    #     "atomic_numbers": [1, 6, 7, 8, 9, 15, 16, 17],
    #     "correlation": 2,
    #     "gate": torch.nn.SiLU(),
    # }
    
    # hparam_dict = {}

    # raveled_model = RaveledModel(**hparam_dict)
    # unraveled_model = GradientsOut(UnraveledModel(**hparam_dict), targets=FORCE_KEY)
    # compare_model_dimensions(raveled_model, unraveled_model, loader, device='cpu')


    # raveled_model, unraveled_model = test_both_models(hparam_dict, loader, device='cpu')


    test_training(
                hparam_dict,
                loader,
                device='cuda')