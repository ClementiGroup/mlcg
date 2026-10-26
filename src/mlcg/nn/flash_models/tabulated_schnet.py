import copy

import torch
from torch_geometric.utils import scatter

from ...data.atomic_data import AtomicData, ENERGY_KEY
from ..schnet import SchNet
from ..kernels import RadiusCSR, build_filter_table, tabulated_cfconv

DEFAULT_NUM_KNOTS = 2048
TABLE_TOLERANCE = 1e-4


class TabulatedCFConv(torch.nn.Module):
    r"""Continuous filter convolution whose filter
    :math:`\phi(d) = \mathrm{cutoff}(d)\,\mathrm{filter}(\mathrm{rbf}(d))`
    is frozen into a cubic Hermite table, see :func:`tabulated_cfconv`.
    """

    def __init__(self, conv, rbf_layer, cutoff_upper: float, num_knots: int):
        super().__init__()
        self.lin1 = copy.deepcopy(conv.lin1)
        self.lin2 = copy.deepcopy(conv.lin2)
        self.cutoff_upper = cutoff_upper
        table, self.table_error = build_filter_table(
            conv.filter_network, rbf_layer, conv.cutoff, cutoff_upper, num_knots
        )
        self.register_buffer("table", table)

    def forward(self, x, pos, row_ptr, col):
        x = tabulated_cfconv(
            self.lin1(x), pos, row_ptr, col, self.table, self.cutoff_upper
        )
        return self.lin2(x)


class TabulatedInteractionBlock(torch.nn.Module):
    def __init__(self, block, rbf_layer, cutoff_upper: float, num_knots: int):
        super().__init__()
        self.conv = TabulatedCFConv(
            block.conv, rbf_layer, cutoff_upper, num_knots
        )
        self.activation = copy.deepcopy(block.activation)
        self.lin = copy.deepcopy(block.lin)

    def forward(self, x, pos, row_ptr, col):
        return self.lin(self.activation(self.conv(x, pos, row_ptr, col)))


class TabulatedSchNet(torch.nn.Module):
    r"""Inference-only SchNet for simulations: every interaction filter is
    tabulated once at conversion (float64, cubic Hermite), so a step
    evaluates no radial basis and no filter network, materializes no
    per-edge tensor, and gets forces from :func:`tabulated_cfconv`'s
    backward. Built from a trained :class:`SchNet` with
    :meth:`flash_from_standard`; the tables are not trainable.

    The radius graph is built by :class:`RadiusCSR` without host
    synchronization (it must be the symmetric graph built here), so a step
    is CUDA-graph capturable when all intra-graph pairs fit its budget.
    Requires ``aggr='add'`` and no periodic boundary conditions.
    """

    def __init__(self, model: SchNet, num_knots: int = DEFAULT_NUM_KNOTS):
        super().__init__()
        self.name = model.name
        self.cutoff_upper = float(model.rbf_layer.cutoff.cutoff_upper)
        self.neighbors = RadiusCSR(self.cutoff_upper)
        self.embedding_layer = copy.deepcopy(model.embedding_layer)
        self.output_network = copy.deepcopy(model.output_network)
        self.interaction_blocks = torch.nn.ModuleList(
            TabulatedInteractionBlock(
                block, model.rbf_layer, self.cutoff_upper, num_knots
            )
            for block in model.interaction_blocks
        )

    @property
    def table_error(self) -> float:
        return max(b.conv.table_error for b in self.interaction_blocks)

    @classmethod
    def flash_from_standard(
        cls, standard_model: SchNet, num_knots: int = DEFAULT_NUM_KNOTS
    ) -> "TabulatedSchNet":
        """Tabulate ``standard_model``; raises ``ValueError`` if it is not a
        supported SchNet or if its filters are not smooth enough for the
        table to reproduce them within ``TABLE_TOLERANCE``."""
        if not isinstance(standard_model, SchNet):
            raise ValueError(f"Expected a SchNet, got {type(standard_model)}")
        aggr = standard_model.interaction_blocks[0].conv.aggr
        if aggr != "add":
            raise ValueError(f"Only aggr='add' can be tabulated, got {aggr}")
        model = cls(standard_model, num_knots)
        if model.table_error > TABLE_TOLERANCE:
            raise ValueError(
                f"Filter table error {model.table_error:.2e} exceeds "
                f"{TABLE_TOLERANCE:.0e} with {num_knots} knots"
            )
        return model

    @staticmethod
    def graph_ptr(data: AtomicData) -> torch.Tensor:
        if hasattr(data, "ptr"):
            return data.ptr
        return data.pos.new_tensor([0, data.pos.shape[0]], dtype=torch.long)

    def forward(self, data: AtomicData) -> AtomicData:
        if data.get("cell") is not None:
            raise NotImplementedError(
                "TabulatedSchNet does not support periodic boundary "
                "conditions; convert with tabulate_filters=False"
            )
        row_ptr, col, valid = self.neighbors(
            data.pos.detach(), data.batch, self.graph_ptr(data)
        )
        x = self.embedding_layer(data.atom_types)
        for block in self.interaction_blocks:
            x = x + block(x, data.pos, row_ptr, col)
        energy = self.output_network(x, data) * valid
        num_graphs = data.ptr.numel() - 1 if hasattr(data, "ptr") else None
        energy = scatter(
            energy, data.batch, dim=0, reduce="sum", dim_size=num_graphs
        )
        data.out.setdefault(self.name, {}).update(
            {ENERGY_KEY: energy.flatten()}
        )
        return data
