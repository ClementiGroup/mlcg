"""
Implementation of `Sharpness-Aware Minimization for Efficiently Improving Generalization <https://arxiv.org/abs/2010.01412>`_
and `ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks <https://arxiv.org/abs/2102.11600>`_.

Adapted from https://github.com/davda54/sam.

It is currently not compatible with PL scheduler system.
"""

import torch
from typing import Callable


class SAM(torch.optim.Optimizer):
    def __init__(
        self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Estimates epsilon parameter/weight values around the neighborhood of
        current weights using an L2 gradient norm scale and the element-wise scaling normalization operator, adding the result to the current parameter/weight values.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Resets all optimizable parameter/weight data values to their
        original state before the epsilon climbing that was done in the first
        step. The base optimizer then takes a step using the original data
        values together with the epsilon-perturbed gradients.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p][
                    "old_p"
                ]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """Compute the two-part full step of SAM optimization. Note that an
        appropriate closure must be passed to the model that zeroes the
        gradient, forwards through the model, computes the loss, and returns
        the loss. For more information, see:

        https://pytorch.org/docs/stable/optim.html#optimizer-step-closure
        """
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        """Computes the total L2 gradient norm accumulated across all model parameter groups"""
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        aa = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if group["adaptive"]:
                        fff = torch.abs(p)
                    else:
                        fff = 1.0

                    aaa = (fff * p.grad).norm(p=2).to(shared_device)
                    aa.append(aaa)
        if len(aa) == 0:
            aa.append(torch.zeros(1))
        norm = torch.norm(torch.stack(aa), p=2)
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
