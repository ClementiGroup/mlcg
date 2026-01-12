"""
Additional callbacks for model training and evaluations
"""
from copy import deepcopy
from pytorch_lightning.callbacks.callback import Callback
from dataclasses import dataclass
import h5py
import torch
from pathlib import Path


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
    def from_batch(cls, batch_data, 
                         save_energy: bool=True,
                         save_forces: bool=True,
                         save_energy_components: bool=True,
                         save_force_components: bool=True,
                         energy_components: list[str]|str|None = None,
                         force_components: list[str]|str|None = None
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
        Each key will correspond to a batch (group of conformations),
        indexed by batch_id.
        """
        # Group data by unique conformations using batch_indices
        batch_indices_np = self.batch_indices.cpu().numpy()
        unique_indices = torch.unique(self.batch_indices)
        
        scattered_data = {}
        
        for conf_idx in unique_indices:
            conf_idx_val = conf_idx.item()
            # Get mask for atoms belonging to this conformation
            mask = self.batch_indices == conf_idx
            
            # Extract data for this conformation
            conf_data = {
                'embeddings': self.embeddings[mask].cpu().numpy(),
                'coordinates': self.coordinates[mask].cpu().numpy(),
            }
            
            if self.energies is not None:
                # Energy is per conformation, not per atom
                conf_data['energies'] = self.energies[conf_idx_val].cpu().numpy()
            
            if self.forces is not None:
                conf_data['forces'] = self.forces[mask].cpu().numpy()
            
            if self.energy_components is not None:
                conf_data['energy_components'] = {}
                for comp_name, comp_values in self.energy_components.items():
                    conf_data['energy_components'][comp_name] = comp_values[conf_idx_val].cpu().numpy()
            
            if self.force_components is not None:
                conf_data['force_components'] = {}
                for comp_name, comp_values in self.force_components.items():
                    conf_data['force_components'][comp_name] = comp_values[mask].cpu().numpy()
            
            scattered_data[conf_idx_val] = conf_data
        
        return scattered_data


class DataSavingCallback(Callback):
    """
    Callback allows to evaluate a given model on a training dataset and save 
    the results. 
    """
    def __init__(self, output_path: str,
                        save_energy: bool=True,
                        save_forces: bool=True, 
                        save_energy_components: bool=False,
                        save_force_components: bool=False, 
                        energy_components: list[str]|str|None = None,
                        force_components: list[str]|str|None = None,
                        cache_size_limit: int=1000
                        ):
        
        self.output_path = Path(output_path)
        self.save_energy = save_energy
        self.save_forces = save_forces
        self.save_energy_components = save_energy_components
        self.save_force_components = save_force_components
        self.energy_components = energy_components if isinstance(energy_components, list) else [energy_components] if energy_components is not None else []
        self.force_components = force_components if isinstance(force_components, list) else [force_components] if force_components is not None else []
        self._validate_input()
        self.cache_size_limit = cache_size_limit  # Number of samples to keep in cache before writing to dataset.

        self.train_cache = []
        self.val_cache = []
        self.current_epoch = 0

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

    def _get_epoch_filename(self, epoch: int, split: str):
        """
        Generate filename for a specific epoch and data split.
        """
        return self.output_path / f"epoch_{epoch}_{split}.h5"

    def save_cache_to_dataset(self, cache: list, split: str, epoch: int):
        """
        Write the cached information to the dataset. Make sure that
        writing is done by a single process only. Need to scatter all the 
        components into different fields.
        """
        if not cache:
            return
        
        filepath = self._get_epoch_filename(epoch, split)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filepath, 'a') as f:
            # Create split group if it doesn't exist
            if split not in f:
                split_group = f.create_group(split)
            else:
                split_group = f[split]
            
            # Process each batch in cache
            for batch_data_instance in cache:
                scattered_data = batch_data_instance.to_dict()
                batch_id = batch_data_instance.batch_ids[0]
                
                # Create group for this batch
                batch_group_name = f"batch_{batch_id}"
                if batch_group_name in split_group:
                    batch_group = split_group[batch_group_name]
                else:
                    batch_group = split_group.create_group(batch_group_name)
                
                # Save data for each conformation in this batch
                for conf_idx, conf_data in scattered_data.items():
                    conf_group_name = f"conf_{conf_idx}"
                    if conf_group_name in batch_group:
                        conf_group = batch_group[conf_group_name]
                    else:
                        conf_group = batch_group.create_group(conf_group_name)
                    
                    # Save embeddings
                    if 'embeddings' not in conf_group:
                        conf_group.create_dataset('embeddings', data=conf_data['embeddings'][None, ...], 
                                                maxshape=(None, *conf_data['embeddings'].shape), 
                                                chunks=True, compression='gzip')
                    else:
                        dataset = conf_group['embeddings']
                        dataset.resize((dataset.shape[0] + 1, *dataset.shape[1:]))
                        dataset[-1] = conf_data['embeddings']
                    
                    # Save coordinates
                    if 'coordinates' not in conf_group:
                        conf_group.create_dataset('coordinates', data=conf_data['coordinates'][None, ...], 
                                                maxshape=(None, *conf_data['coordinates'].shape), 
                                                chunks=True, compression='gzip')
                    else:
                        dataset = conf_group['coordinates']
                        dataset.resize((dataset.shape[0] + 1, *dataset.shape[1:]))
                        dataset[-1] = conf_data['coordinates']
                    
                    # Save energies
                    if self.save_energy and conf_data.get('energies') is not None:
                        if 'energies' not in conf_group:
                            conf_group.create_dataset('energies', data=conf_data['energies'][None], 
                                                    maxshape=(None,), 
                                                    chunks=True, compression='gzip')
                        else:
                            dataset = conf_group['energies']
                            dataset.resize((dataset.shape[0] + 1,))
                            dataset[-1] = conf_data['energies']
                    
                    # Save forces
                    if self.save_forces and conf_data.get('forces') is not None:
                        if 'forces' not in conf_group:
                            conf_group.create_dataset('forces', data=conf_data['forces'][None, ...], 
                                                    maxshape=(None, *conf_data['forces'].shape), 
                                                    chunks=True, compression='gzip')
                        else:
                            dataset = conf_group['forces']
                            dataset.resize((dataset.shape[0] + 1, *dataset.shape[1:]))
                            dataset[-1] = conf_data['forces']
                    
                    # Save energy components
                    if self.save_energy_components and conf_data.get('energy_components'):
                        if 'energy_components' not in conf_group:
                            energy_comp_group = conf_group.create_group('energy_components')
                        else:
                            energy_comp_group = conf_group['energy_components']
                        
                        for comp_name, comp_value in conf_data['energy_components'].items():
                            if comp_name not in energy_comp_group:
                                energy_comp_group.create_dataset(comp_name, data=comp_value[None], 
                                                                maxshape=(None,), 
                                                                chunks=True, compression='gzip')
                            else:
                                dataset = energy_comp_group[comp_name]
                                dataset.resize((dataset.shape[0] + 1,))
                                dataset[-1] = comp_value
                    
                    # Save force components
                    if self.save_force_components and conf_data.get('force_components'):
                        if 'force_components' not in conf_group:
                            force_comp_group = conf_group.create_group('force_components')
                        else:
                            force_comp_group = conf_group['force_components']
                        
                        for comp_name, comp_value in conf_data['force_components'].items():
                            if comp_name not in force_comp_group:
                                force_comp_group.create_dataset(comp_name, data=comp_value[None, ...], 
                                                                maxshape=(None, *comp_value.shape), 
                                                                chunks=True, compression='gzip')
                            else:
                                dataset = force_comp_group[comp_name]
                                dataset.resize((dataset.shape[0] + 1, *dataset.shape[1:]))
                                dataset[-1] = comp_value

    def empty_cache(self, split: str):
        """
        Empty the cache for a specific split.
        """
        if split == 'train':
            self.train_cache = []
        elif split == 'val':
            self.val_cache = []

    def on_fit_start(self, trainer, pl_module):
        """
        Initialize the output directory and set up for data collection.
        """
        if trainer.is_global_zero:
            self.output_path.mkdir(parents=True, exist_ok=True)
            print(f"DataSavingCallback: Output directory created at {self.output_path}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch_data, batch_idx):
        """
        Save all the information of interest: embeddings, coordinates, forces to cache.
        If cache is large, save it to dataset and clear the cache.
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

        self.train_cache.append(batch_data_instance)

        # Save periodically during epoch if cache gets too large
        if len(self.train_cache) >= self.cache_size_limit and trainer.is_global_zero:
            self.save_cache_to_dataset(self.train_cache, 'train', self.current_epoch)
            self.empty_cache('train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch_data, batch_idx, dataloader_idx=0):
        """
        Save validation batch data to cache.
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

        self.val_cache.append(batch_data_instance)

        # Save periodically during epoch if cache gets too large
        if len(self.val_cache) >= self.cache_size_limit and trainer.is_global_zero:
            self.save_cache_to_dataset(self.val_cache, 'val', self.current_epoch)
            self.empty_cache('val')

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Save remaining training data at the end of the epoch.
        """
        if trainer.is_global_zero and self.train_cache:
            self.save_cache_to_dataset(self.train_cache, 'train', self.current_epoch)
            self.empty_cache('train')

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Save remaining validation data at the end of the validation epoch.
        """
        if trainer.is_global_zero and self.val_cache:
            self.save_cache_to_dataset(self.val_cache, 'val', self.current_epoch)
            self.empty_cache('val')
        
        # Increment epoch counter
        self.current_epoch += 1


if __name__ == "__main__":
    callback = DataSavingCallback(
        output_path="./output_data",
        save_energy=True,
        save_forces=True,
        save_energy_components=False
    )
    print(callback)
