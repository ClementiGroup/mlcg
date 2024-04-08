"""Adapts the schnet models that were saved with previous versions of 
pygs (Pytorch Geometric) and make them compatible with a newer version
in use.
"""

import warnings

import torch
import mlcg.nn.schnet
from mlcg.nn.schnet import CFConv, SchNet


def get_refreshed_cfconv_layer(old_cfconv: CFConv):
    """Extract the weights and reinstantiate the `CFConv` object
    such as to make it compatible with the current pyg.

    Parameters
    ----------
    old_cfconv: the `CFConv` from deserialized SchNet checkpoints
        possibly saved with a previous versions of pyg.
    """
    # Tensor was once imported as a standalone symbol in an older
    # version of mlcg-tools (or pyg) but not anymore, thus we have
    # to monkey patch it here in order for some older checkpoints
    # to work
    mlcg.nn.schnet.Tensor = torch.Tensor
    # extract init args
    filter_network = old_cfconv.filter_network
    cutoff = old_cfconv.cutoff
    in_channels = old_cfconv.lin1.in_features
    out_channels = old_cfconv.lin2.out_features
    num_filters = old_cfconv.lin1.out_features
    aggr = old_cfconv.aggr
    # extract the state dict
    # clone is used here, since the state_dict of filter_network
    # will be overwritten by the __init__ of `new_cfconv`
    state_dict = {k: v.clone() for k, v in old_cfconv.state_dict().items()}
    device = next(iter(state_dict.values())).device
    # rebuild
    new_cfconv = CFConv(
        filter_network=filter_network,
        cutoff=cutoff,
        in_channels=in_channels,
        out_channels=out_channels,
        num_filters=num_filters,
        aggr=aggr,
    ).to(device)
    new_cfconv.load_state_dict(state_dict)
    return new_cfconv


def _search_for_schnet(top_level: torch.nn.Module):
    """Recursively search for SchNet in all submodules."""
    if isinstance(top_level, SchNet):
        yield top_level
    elif isinstance(top_level, torch.nn.ModuleList):
        for module in top_level:
            yield from _search_for_schnet(module)
    elif isinstance(top_level, torch.nn.ModuleDict):
        for module in top_level.values():
            yield from _search_for_schnet(module)
    else:
        # e.g., a `SumOut` or `GradientsOut`
        for module in top_level.children():
            yield from _search_for_schnet(module)


def refresh_module_with_schnet_(schnet_containing: torch.nn.Module):
    """In-place refresh all cfconv_layers in a torch.nn.Module that possibly
    contains a `mlcg.nn.SchNet` inside. See `get_refreshed_cfconv_layer` for
    what this refreshing is exactly doing.

    Parameters
    ----------
    schnet_containing (torch.nn.Module): a module that is expected to have
    a SchNet as submodule.
    """
    all_schnets = list(_search_for_schnet(schnet_containing))
    if len(all_schnets) == 0:
        raise ValueError(
            f"No SchNet has been found as a submodule of the input "
            f"{schnet_containing}"
        )
    if len(all_schnets) > 1:
        warnings.warn(
            f"{len(all_schnets)} SchNet has been found as a submodule "
            f"of the input {schnet_containing}, please ensure whether "
            f"this is expected."
        )
    for schnet in all_schnets:
        for ib in schnet.interaction_blocks:
            ib.conv = get_refreshed_cfconv_layer(ib.conv)


def load_and_adapt_old_checkpoint(f, **kwargs):
    """Load and adapt an older checkpoint from training with a previous
    version of pyg. Using this function instead of `torch.load` can bypass
    the import error caused by an newer version of pyg, and automatically
    adapts the SchNet to make it usable.

    Parameters
    ----------
    f (Union[str, PathLike, BinaryIO, IO[bytes]]): the path or file-like object
        of a checkpoint from a possibly older version of pyg.
    kwargs: see the docstring of `torch.load` for details.
    """
    import sys
    import torch_geometric
    from packaging import version

    monkey_patched = False
    try:
        if version.parse(torch_geometric.__version__) >= version.parse("2.5"):
            # monkey patch for the inspector.py, which has been moved to another place
            # in recent pygs
            sys.modules["torch_geometric.nn.conv.utils.inspector"] = (
                torch_geometric.inspector
            )
            monkey_patched = True
        module = torch.load(f, **kwargs)
        refresh_module_with_schnet_(module)
    finally:
        if monkey_patched:
            # recover changes made by the monkey patch
            del sys.modules["torch_geometric.nn.conv.utils.inspector"]
    return module
