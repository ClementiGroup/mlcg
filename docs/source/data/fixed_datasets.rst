Fixed datasets
========

``mlcg.datasets`` contains a template ``InMemoryDataset`` for CLN025, a 10 amino acid long mini protein that shows prototypical folding and unfolding behavior. The ``ChignolinDataset`` class illustrates how a general dataset can be downloaded, unpacked/organized, transformed/processed, and collated for training models. `Here <https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html>`_, users can find more information on implementing custom datasets.

Alanine Dipeptide Dataset
-------------------------
Dataset of a single 1M step trajectory of alanine dipeptide in explicit water. The trajectory is simulated using a Langevin scheme with [ACEMD]_ at 300K through the [AMBER_ff_99SB_ILDN]_ force force field. The cubic simulation box was 2.3222 cubic nm, an integration timestep of 2 fs was used, the solvent was composed of 651 [TIP3P]_ water molecules, electrostatics were computed every two steps using the PME method with a real-space cutoff of 9 nm and a grid spacing of 0.1 nm, and all bonds between heavy and hydrogen atoms were constrained.

.. autoclass:: mlcg.datasets.alanine_dipeptide.AlanineDataset
        :members:

Chignolin Dataset
-----------------

Dataset of 3744 individual all-atom trajectories simulated using [ACEMD]_ using the adaptive sampling strategy described in [AdaptiveStrategy]_ . All trajectories were simulated at 350K with [CHARM22Star]_ in a cubic box of 40 angstroms :sup:``3`` with 1881 TIP3P water molecules and two Na :sup:``+`` ions using a Langevin integrator with an integration timestep of 4 fs, a damping constant of 0.1 :sup:``-1`` ps, heavy-hydrogen constraints, and a PME cutoff of 9 angstroms and a PME mesh grid of 1 angstrom. The total aggregate simulation time is 187.2 us. 

.. autoclass:: mlcg.datasets.chignolin.ChignolinDataset
        :members: