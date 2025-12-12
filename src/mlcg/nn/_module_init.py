import torch


def init_xavier_uniform(
    module: torch.nn.Module, zero_bias: bool = True
) -> None:
    r"""initialize (in place) weights of the input module using xavier uniform.
    Works only on `torch.nn.Linear` at the moment and the bias are set to 0
    by default.

    Parameters
    ----------
    module:
        a torch module
    zero_bias:
        If True, the bias will be filled with zeroes. If False,
        the bias will be filled according to the torch.nn.Linear
        default distribution:

        .. math:

            \text{Uniform}(-\sqrt{k}, sqrt{k}) ; \quad \quad k = \frac{1}{\text{in_features}}

    """
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None and zero_bias == True:
            torch.nn.init.constant_(module.bias, 0.0)
