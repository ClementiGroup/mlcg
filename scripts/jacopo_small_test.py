import torch

import sys
sys.path.insert(0, "/local_scratch2/jacopo/software/mlcg")

from mlcg.nn import ForceMSE
from mlcg.nn.gradients import GradientsOut
from mlcg.data._keys import FORCE_KEY

# from allegro_model import StandardAllegro as Model
from mlcg.nn.allegro import StandardAllegro as Model

from rangemp.datasets import AQMgasDataset
from mlcg.pl.data import DataModule
from mlcg.nn import Loss
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

DATASET_PATH = "/srv/data/kamenrur95/datasets/AQMgas/mlcg/"
DATASET_SPLITS = "/srv/data/kamenrur95/datasets/AQMgas/mlcg/selection_n_atoms_2_train_0.85_val_0.1_seed_4272389.npz"

dataset = AQMgasDataset(DATASET_PATH)

loader = DataModule(
                dataset,
                splits=DATASET_SPLITS,
                loading_stride=1,
                batch_size=32,
                inference_batch_size=32,
                num_workers=6,
                save_local_copy="False",
                pin_memory="True",
                log_dir='.'
                )
loader.load_dataset()
loader.setup(stage="test")  # 499 strutture


def test_training(r_max,
                  loader,
                  device='cpu'):

    

    test_mace = Model(
        r_max=r_max)
    # print(test_mace)
    model = test_mace
    # model = GradientsOut(test_mace, targets=FORCE_KEY)
    # print(model)
    # model = EnergyOut(test_schnet, targets=['energy'])
    loss_fn = Loss([
        ForceMSE(force_kwd='forces'),
        ForceMSE(force_kwd='energy'),
    ],
    weights=[
        1.0, 0.01
    ])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Test one epoch training
    # print("____________________")
    model.train()
    model.to(device)

    start = time.time()
    for epoch in tqdm(range(20)):
        losses = []
        for data in loader.test_dataloader():
            data.to(device)
            optimizer.zero_grad()
            data = model(data)
            data.out.update(**data.out[model.name])
            data.out['forces'].to(data.out['energy'].dtype)
            loss = loss_fn(data)
            losses.append(loss)
            loss.backward()
            optimizer.step()
        # print(torch.tensor(losses).mean())
    end = time.time()
    
    print(f"Train time 20 epochs {end-start}")
    print("-------------------------------------------")

    # torch.save(data.pos, "positions.dat")
    # torch.save(data.atom_types, "atom_typ.dat")
    # torch.save(data.n_atoms, "n_atoms.dat")


if __name__ == "__main__":
    r_max = 5.0
    


    test_training(
                r_max,
                loader,
                device='cuda')
            


# uv pip install --extra-index-url=https://download.pytorch.org/whl/cu124 torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0
# uv pip install torch_geometric
# uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
# uv pip install lightning tensorboard torchtnt