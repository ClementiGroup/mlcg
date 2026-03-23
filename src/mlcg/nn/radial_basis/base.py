import torch


class _RadialBasis(torch.nn.Module):
    r"""Abstract radial basis function class"""

    def __init__(self):
        super(_RadialBasis, self).__init__()
        self.cutoff = None

    def check_cutoff(self):
        if self.cutoff.cutoff_upper < self.cutoff.cutoff_lower:
            raise ValueError(
                "Upper cutoff {} is less than lower cutoff {}".format(
                    self.cutoff.cutoff_upper, self.cutoff.cutoff_lower
                )
            )

    def plot(self):
        r"""Method for quickly visualizing a specific basis. This is useful for
        inspecting the distance coverage of basis functions for non-default lower
        and upper cutoffs.
        """

        import matplotlib.pyplot as plt

        distances = torch.linspace(
            self.cutoff.cutoff_lower - 1,
            self.cutoff.cutoff_upper + 1,
            1000,
        )
        expanded_distances = self(distances)

        for i in range(expanded_distances.shape[-1]):
            plt.plot(
                distances.numpy(), expanded_distances[:, i].detach().numpy()
            )
        plt.show()

    def forward(self):
        raise NotImplementedError
