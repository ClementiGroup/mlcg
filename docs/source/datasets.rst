Datasets
========

``mlcg.datasets`` contains a template ``InMemoryDataset`` for CLN025, a 10 amino acid long mini protein that shows prototypical folding and unfolding behavior. The ``ChignolinDataset`` class illustrates how a general dataset can be downloaded, unpacked/organized, transformed/processed, and collated for training models. `Here <https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html>`_, users can find more information on implementing custom datasets.

Chignolin Dataset
-----------------

Dataset of 3744 individual all-atom trajectories simulated using [ACEMD]_ using the adaptive sampling strategy described in [AdaptiveStrategy]_ . All trajectories were simulated at 350K with [CHARM22Star]_ in a cubic box of 40 angstroms :sup:``3`` with 1881 TIP3P water molecules and two Na :sup:``+`` ions using a Langevin integrator with an integration timestep of 4 fs, a damping constant of 0.1 :sup:``-1`` ps, heavy-hydrogen constraints, and a PME cutoff of 9 angstroms and a PME mesh grid of 1 angstrom. The total aggregate simulation time is 187.2 us. 

.. autoclass:: mlcg.datasets.chignolin.ChignolinDataset

Alanine Dipeptide Dataset
-------------------------
Dataset of a single 1M step trajectory of alanine dipeptide in explicit water. The trajectory is simulated using a Langevin scheme with [ACEMD]_ at 300K through the [AMBER_ff_99SB_ILDN]_ force force field. The cubic simulation box was 2.3222 cubic nm, an integration timestep of 2 fs was used, the solvent was composed of 651 [TIP3P]_ water molecules, electrostatics were computed every two steps using the PME method with a real-space cutoff of 9 nm and a grid spacing of 0.1 nm, and all bonds between heavy and hydrogen atoms were constrained.

.. autoclass:: mlcg.datasets.alanine_dipeptide.AlanineDataset

Custom H5 Dataset
-----------------

Users may assemble their own curated dataset using an H5 format. This allows for the possiblity of training on multiple types of molecules or data from different system conditiions.

Inroduction
^^^^^^^^^^^

HDF5 format benefits the dataset management for mlcg when training/validation involves multiple molecules of vastly different lengths and when parallelization is used.
The main features are:

1. The internal structure mimics the hierarchy of the dataset itself, such that we don't have to replicate it on filesystem.
2. we don't have to actively open all files in the process
3. we can transparently load only the necessary part of the dataset to the memory

This file contains the python data structures for handling the HDF5 file and its content, i.e., the coordinates, forces and embedding vectors for multiple CG molecules.
An example HDF5 file structure and correponding class types after loading:

.. code-block::

        / (HDF-group, => ``H5Dataset._h5_root``)
        ├── OPEP (HDF-group =(part, according to "partition_options")=> ``Metaset`` in a ``Partition``)
        │   ├── opep_0000 (HDF-group, => ``MolData``)
        │   │   ├── attrs (HDF-attributes of the molecule "/OPEP/opep_0000")
        │   │   │   ├── cg_embeds (a 1-d numpy.int array)
        │   │   │   ├── N_frames (int, number of frames = size of cg_coords on axis 0)
        │   │   │   ... (optional, e.g., cg_exc_pairs cg_top, cg_pdb, etc that corrsponds to the molecule)
        │   │   ├── cg_coords (HDF-dataset of the molecule "/OPEP/opep_0000", 3-d numpy.float32 array)
        │   │   └── cg(_delta)_forces (HDF-dataset of the molecule "/OPEP/opep_0000", 3-d numpy.float32 array)
        │   ... (other molecules in "/OPEP")
        ├── CATH (HDF-group ="partition_options"=> ``Metaset`` in a ``Partition``)
        ...

Basic bricks: ``MolData`` and ``MetaSet``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Data structure ``MolData`` is the basic brick of the dataset.
It holds embeds, coords and forces of certain number of frames for a single molecule.
.sub_sample: method for performing indexing on the frames.


.. autoclass:: mlcg.datasets.MolData


Data strcuture ``Metaset`` holds multiple molecules and joins their frame indices together.
One can access frames of underlying MolData objects with a continuous index.
The idea is that the molecules in a ``Metaset`` have similar numbers of CG beads, such that the sample requires similar processing time when passing through a neural network.
When this rule is enforced, it will allow automatic balancing of batch composition during training and validation.



* ``.create_from_hdf5_group``: initiate a Metaset by loading data from given HDF-group.mol_list, detailed_indices, hdf_key_mapping, stride and parallel can control which subset is loaded to the Metaset (details see the description of "H5Dataset")
* ``.trim_down_to``: drop frames of underlying for obtaining a subset with desired number of frames. The indices of frames to be dropped are controlled by a random number generator. When the parameter "random_seed" and the MolData order and number of frames are the same, the frames to be dropped will be reproducible.
* ``len() (__len__)``: return total number of samples
* ``[] (__get_item__)``: infer the molecule and get the corresponding frame according to the given unified index. Return value is grouped by an ``AtomicData`` object.

.. autoclass:: mlcg.datasets.MetaSet

Train test split: ``Partition``
^^^^^^^^^^^^^^^^^^^^^^^^

Data structure ``Partition`` can hold multiple Metaset for training and validation purposes.
Its main function is to automatic adjust (subsample) the Metaset(s) and the underlying MolData to form a balanced data source, from which a certain number of batches can be drawn. Each batch will contain a pre-given number of samples from each Metaset.
One or several ``Partition``s can be created to load part of the same HDF5 file into the memory.
The exact content inside a ``Partition`` object is controlled by the ``partition_options`` as a parameter for initializing a ``H5Dataset``.

.. autoclass:: mlcg.datasets.Partition


Full Dataset: ``H5Dataset``
^^^^^^^^^^^^^^^^^^^^^^^^


Data strcuture ``H5Dataset`` opens a HDF5 file and establish the partitions.
``partition_options`` describe what partitions to create and what content to load into them. (Detailed description and examples are as followed.)
``loading_options`` mainly deal with the HDF key mapping (which datasets/attributes corresponds to the coordinates, forces and embeddings) as well as (optionally) the information for correctly split the dataset in a parallel training/validation.

An example "partition_options" (as a Python Mappable (e.g., dict)):

.. code-block::

        {
                "train": {
                        "metasets": {
                                "OPEP": {
                                        "molecules": [
                                                "opep_0000",
                                                "opep_0001",
                                                ...
                                        ],
                                        "stride": 1, # optional, default 1
                                        "detailed_indices": {
                        # optional, providing the indices of frames to work with (before striding and splitting for parallel processes).
                                                # optional,
                            "opep_0000":
                                val_ratio: 0.1
                                test_ratio: 0.1
                                seed: 12345
                            "opep_0001":
                                val_ratio: 0.1
                                test_ratio: 0.1
                                seed: 12345
                            "filename": ./splits

                        # If detailed_indices are not provided for a given molecule, then it is equivalent to np.arange(N_frames)
                                                    "opep_0000": [1, 3, 5, 7, 9, ...],


                                        },
                                },
                                "CATH": {
                                        "molecules": [
                                                "cath_1b43A02",
                                                ...
                                        ],
                                        "stride": 1, # optional
                                        "detailed_indices": {}, # optional
                                }
                        },
                        "batch_size": {
                                # each batch will contain 24 samples from
                                # The two metasets will be trimmed down according to this ratio
                                # so optimally it should be proportional to the ratio of numbers of frames in the metasets.
                                "OPEP": 24,
                                "CATH": 6,
                        },
                        "subsample_random_seed": 42, # optional, default 42. Random seed for trimming down the frames.
                        "max_epoch_samples": None, # optional, default None. For setting a desired dataset size.
                        # Works by subsampling the dataset after it is loaded with the given stride.
                },
                "val": { # similar
                }
        }

An example "loading_options" (as a Python Mappable (e.g., dict)):

.. code-block::

        {
                "hdf_key_mapping": {
                        # keys for reading CG data from HDF5 groups
                        "embeds": "attrs:cg_embeds",
                        "coords": "cg_coords",
                        "forces": "cg_delta_forces",
                },
                "parallel": { # optional, default rank=0 and world_size=1 (single process).
                        # For split the dataset evenly and load only the necessary parts to each process in a parallelized train/val setup
                        "rank": 0, # which process is this
                        "world_size": 1, # how many processes are there
                }
        }

.. autoclass:: mlcg.datasets.H5Dataset


Loading into PyTorch: ``H5PartitionDataLoader``
^^^^^^^^^^^^^^^^^^^^^^^^

Class ``H5PartitionDataLoader`` mimics the pytorch data loader and can generate training/validation batches with fixed proportion of samples from several Metasets in a Partition.
The proportion and batch size is defined when the partition is initialized.
When the molecules in each Metaset have similar embedding vector lengths, the processing of output batches will require a more or less fixed size of VRAM on GPU, which can benefit the

Note:
1. Usually in a train-val split, each molecule goes to either the train or the test partition.
     In some special cases (e.g., non-transferable training) one molecule can be part of both partitions.
     In this situation, "detailed_indices" can be set to assign the corresponding frames to the desired partitions.
In addition one can pass in detailed_indices as a dictionary to split frames based on training/test (see partition_options example above)

.. autoclass:: mlcg.datasets.H5PartitionDataLoader



