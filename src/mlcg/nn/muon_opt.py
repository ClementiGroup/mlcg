import torch
from torch.optim import AdamW


class AutoMuon(torch.optim.Optimizer):
    """
    Muon wrapper that automatically routes parameters:
      - 2D+ weights whose name doesn't match muon_exclude_names → Muon
      - everything else (biases, embeddings, norms, etc.) → AdamW

    Use with MuonCLI, which supplies named_parameters at construction time.
    When instantiated by Lightning CLI directly (before MuonCLI intercepts),
    params is treated as plain parameters and all go to AdamW as a safe fallback.
    """

    def __init__(
        self,
        params,
        muon_exclude_names: tuple = ("embed", "norm", "bn"),
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple = (0.9, 0.95),
        adamw_weight_decay: float = 0.01,
    ):
        self._muon_kwargs = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            nesterov=nesterov, ns_steps=ns_steps,
        )
        self._adamw_kwargs = dict(
            lr=adamw_lr, betas=tuple(adamw_betas), weight_decay=adamw_weight_decay,
        )
        self._muon_exclude_names = muon_exclude_names

        # params is either an iterable of plain tensors (CLI first-pass)
        # or (name, tensor) tuples (MuonCLI re-instantiation)
        params = list(params)
        if params and isinstance(params[0], tuple):
            named = params  # already (name, param) pairs
        else:
            named = [(f"param_{i}", p) for i, p in enumerate(params)]

        muon_params, adamw_params = [], []
        muon_names, adamw_names = [], []
        for name, param in named:
            if (
                param.ndim >= 2
                and not any(s in name for s in self._muon_exclude_names)
            ):
                muon_params.append(param)
                muon_names.append(name)
            else:
                adamw_params.append(param)
                adamw_names.append(name)
        print("WARNING: Muon optimizer selected, will only optimize 2D weight params")
        print("Muon params:")
        print(muon_names)
        print("AdamW params")
        print(adamw_names)
        self._muon = torch.optim.Muon(muon_params, **self._muon_kwargs)
        self._adamw = AdamW(adamw_params, **self._adamw_kwargs)
        defaults = dict(lr=lr)
        super().__init__([p for _, p in named], defaults)
        self.param_groups = self._muon.param_groups + self._adamw.param_groups

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._muon.step()
        self._adamw.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self._muon.zero_grad(set_to_none=set_to_none)
        self._adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"muon": self._muon.state_dict(), "adamw": self._adamw.state_dict()}

    def load_state_dict(self, state_dict):
        self._muon.load_state_dict(state_dict["muon"])
        self._adamw.load_state_dict(state_dict["adamw"])