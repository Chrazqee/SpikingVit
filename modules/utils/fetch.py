from omegaconf import DictConfig

from modules.data.genx import DataModule as genx_data_module
from modules.detection_module import Module as det_module

def fetch_data_module(config: DictConfig):
    batch_size_train = config.batch_size.train
    batch_size_eval = config.batch_size.eval
    num_workers_generic = config.hardware.get('num_workers', None)
    num_workers_train = config.hardware.num_workers.get('train', num_workers_generic)
    num_workers_eval = config.hardware.num_workers.get('eval', num_workers_generic)
    dataset_str = config.dataset.name
    if dataset_str in {'gen1', 'gen4'}:
        dataset_config = config.dataset
        return genx_data_module(dataset_config,
                                num_workers_train=num_workers_train,
                                num_workers_eval=num_workers_eval,
                                batch_size_train=batch_size_train,
                                batch_size_eval=batch_size_eval)
    raise NotImplementedError

def fetch_model_module(config: DictConfig):
    # not implement judge
    return det_module(config)
