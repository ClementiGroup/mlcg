import torch
from typing import Final, Optional
import numpy as np

try:
    from torchmdnet.models.model import create_model, load_model

except ImportError as e:
    print(e)
    print(
        """Please install or set torchmd-net to your path before using this interface.
    To install you can either run
    'pip install git+https://github.com/felixmusil/torchmd-net.git'
    or clone the repository and add it to PYTHONPATH."""
    )


class TorchMDNetInterface(torch.nn.Module):
    name: Final[str] = "torchmd-net"

    def __init__(self, hparams: dict) -> None:
        super().__init__()
        self.hparams = hparams
        # we don't need prior models from torchmd-net
        self.hparams["prior_model"] = False
        self.model = create_model(self.hparams, None, None, None)
        self.derivative = True

    def forward(self, data):
        ndata = self.data2ndata(data)
        energy, forces = self.model(
            z=ndata.atom_types,
            pos=ndata.pos,
            batch=ndata.batch,
        )
        data.out[self.name] = {
            "energy": energy.flatten(),
            "forces": forces,
        }
        return data

    def data2ndata(self, data):
        return data
