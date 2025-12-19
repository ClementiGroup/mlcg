"""
Additional callbacks for model training and evaluations
"""
from copy import deepcopy
from pytorch_lightning.callbacks.callback import Callback
from dataclasses import dataclass
from mlcg.data.atomic_data import AtomicData


from ..data._keys import (
    ENERGY_KEY,
    FORCE_KEY,
    ATOM_TYPE_KEY
)

@dataclass
class BatchData:
    """
    Class to hold required batch data information. 
    Instances of the class will be put in cache for saving to dataset.
    """
    embeddings: list
    coordinates: list
    energies: list|None = None
    forces: list|None = None
    energy_components: dict|None = None
    force_components: dict|None = None
    batch_indices: list = None
    batch_ids: list = None

    @classmethod
    def from_batch(cls, batch_data: AtomicData, 
                         save_energy: bool=True,
                         save_forces: bool=True,
                         save_energy_components: bool=True,
                         save_force_components: bool=True,
                         energy_components: list[str]|str|None = None,
                         force_components: list[str]|str|None =None
                         ):
        """
        Create a BatchData instance from a batch_data object. 
        """
        embeddings = deepcopy(batch_data.out[ATOM_TYPE_KEY].detach())
        coordinates = deepcopy(batch_data.pos.detach())
        energies = deepcopy(batch_data.out[ENERGY_KEY].detach()) if save_energy else None
        forces = deepcopy(batch_data.out[FORCE_KEY].detach()) if save_forces else None

        energy_comps = {}
        if save_energy_components and energy_components:
            energy_components = energy_components if isinstance(energy_components, list) else [energy_components]
            for comp in energy_components:
                energy_comps[comp] = deepcopy(batch_data.out[comp][ENERGY_KEY].detach())

        force_comps = {}
        if save_force_components and force_components:
            force_components = force_components if isinstance(force_components, list) else [force_components]
            for comp in force_components:
                force_comps[comp] = deepcopy(batch_data.out[comp][FORCE_KEY].detach())

        batch_indices = None  # To be filled later
        batch_ids = None      # To be filled later

        return cls(embeddings=embeddings,
                   coordinates=coordinates,
                   energies=energies,
                   forces=forces,
                   energy_components=energy_comps if energy_comps else None,
                   force_components=force_comps if force_comps else None,
                   batch_indices=[],
                   batch_ids=[]
                                              )
    def to_dict(self):
        """
        Scatter the batch data into a dictionary for saving. 
        Each key will correspond to unique embedding of a single system,
        hashed to be indexable. 
        """
        pass



class DataSavingCallback(Callback):
    """
    Callback allows to evaluate a given model on a training dataset and save 
    the results. 
    """
    def __init__(self, save_energy: bool=True,
                        save_forces: bool=True, 
                        save_energy_components: bool=False,
                        save_force_components: bool=False, 
                        energy_components: list[str]|str|None = None,
                        force_components: list[str]|str|None = None,
                        cache_size_limit: int=1000
                        ):
        
        self.save_energy = save_energy
        self.save_forces = save_forces
        self.save_energy_components = save_energy_components
        self.save_force_components = save_force_components
        self.energy_components = energy_components if isinstance(energy_components, list) else [energy_components] if energy_components is not None else []
        self.force_components = force_components if isinstance(force_components, list) else [force_components] if force_components is not None else []
        self._validate_input()
        self.cache_size_limit = cache_size_limit  # Number of samples to keep in cache before writing to dataset.

        self.cache = []

    def _validate_input(self):
        """
        Validate the input parameters. 
        """
        if self.energy_components and not self.save_energy_components:
            raise ValueError("energy_components specified but save_energy_components is False")
        if self.force_components and not self.save_force_components:
            raise ValueError("force_components specified but save_force_components is False")
        if self.save_energy_components and not self.energy_components:
            raise ValueError("save_energy_components is True but no energy_components specified")
        if self.save_force_components and not self.force_components:
            raise ValueError("save_force_components is True but no force_components specified")

    def save_cache_to_dataset(self):
        """
        Write the cached information to the dataset. Make sure that
        writing is done by a single process only. Need to scatter all the 
        components into different fields 
        """
        pass    

    def empty_cache(self):
        """
        Empty the cache. 
        """
        self.cache = []

    def on_fit_start(self, trainer, pl_module):
        """
        Generate a new dataset, that will be used to store the information. 

        """
        # Get information regarding the dataset. 
        pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch_data, batch_idx):
        """
        Save all the information of interest: embeddings, coordinates, forces to cache.
        If cache is large, save it to dataset and clear the cache. Need to have cache to
        """
        batch_data_instance = BatchData.from_batch(
            batch_data,
            save_energy=self.save_energy,
            save_forces=self.save_forces,
            save_energy_components=self.save_energy_components,
            save_force_components=self.save_force_components,
            energy_components=self.energy_components,
            force_components=self.force_components
        )
        # Fill in batch indices and ids
        batch_data_instance.batch_indices = deepcopy(batch_data.batch.detach())
        batch_data_instance.batch_ids.append(batch_idx)
        print(dir(batch_data))

        self.cache.append(batch_data_instance)

        # Only one process should be saving to dataset.
        if len(self.cache) >= self.cache_size_limit and trainer.is_global_zero:
            self.save_cache_to_dataset()
            self.empty_cache()

if __name__ == "__main__":
    callback = DataSavingCallback(save_energy_components=False)
    print(callback)

