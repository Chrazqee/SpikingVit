import os

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from callbacks.callbacks import get_ckpt_callback, get_viz_callback

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ['HYDRA_FULL_ERROR'] = '1'

import sys

# to solve package relative import problems!
sys.path.append(sys.path[0] + '/models/encoder/layers')

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.strategies import DDPStrategy
import pytorch_lightning as pl

from conf.modifier import dynamically_modify_train_config
from data.utils.types import DatasetSamplingMode
from modules.utils.fetch import fetch_data_module, fetch_model_module


@hydra.main(config_path='conf', config_name='train', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    # ---------------------
    # Reproducibility
    # ---------------------
    dataset_train_sampling = config.dataset.train.sampling
    assert dataset_train_sampling in iter(DatasetSamplingMode)
    disable_seed_everything = dataset_train_sampling in (DatasetSamplingMode.STREAM, DatasetSamplingMode.MIXED)
    if disable_seed_everything:
        print('Disabling PL seed everything because of unresolved issues with shuffling during training on streaming '
              'datasets')

    # ---------------------
    # DDP
    # ---------------------
    gpu_config = config.hardware.gpus
    gpus = OmegaConf.to_container(gpu_config) if OmegaConf.is_config(gpu_config) else gpu_config
    gpus = gpus if isinstance(gpus, list) else [gpus]
    distributed_backend = config.hardware.dist_backend
    assert distributed_backend in ('nccl', 'gloo'), f'{distributed_backend=}'

    strategy = DDPStrategy(process_group_backend=distributed_backend,
                           find_unused_parameters=False,
                           gradient_as_bucket_view=True) if len(gpus) > 1 else None

    # ---------------------
    # Data
    # ---------------------
    data_module = fetch_data_module(config=config)

    # ---------------------
    # Logging and Checkpoints
    # ---------------------
    tb_logger = TensorBoardLogger(save_dir='./simple_down_sample_all_batchnorm',
                                  name='spikingDetection_gen1',
                                  version='____',
                                  sub_dir='seq_length_5',
                                  log_graph=False)

    full_config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    tb_logger.log_hyperparams(full_config_dict)
    # ---------------------
    # Model
    # ---------------------
    # 🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗 `ckpt_path` 🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗
    ckpt_path = None
    module = fetch_model_module(config=config)

    # ---------------------
    # Callbacks and Misc
    # ---------------------
    callbacks = list()
    callbacks.append(get_ckpt_callback())
    if config.training.lr_scheduler.use:
        callbacks.append(
            LearningRateMonitor(logging_interval='step'))

    callbacks.append(get_viz_callback(config=config))
    # ---------------------
    # Training
    # ---------------------
    val_check_interval = config.validation.val_check_interval
    check_val_every_n_epoch = config.validation.check_val_every_n_epoch
    assert val_check_interval is None or check_val_every_n_epoch is None

    trainer = pl.Trainer(
        accelerator='gpu',
        callbacks=callbacks,
        enable_checkpointing=True,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        default_root_dir=None,
        devices=gpus,
        gradient_clip_val=config.training.gradient_clip_val,
        gradient_clip_algorithm='value',
        limit_train_batches=config.training.limit_train_batches,
        limit_val_batches=config.validation.limit_val_batches,
        logger=tb_logger,
        log_every_n_steps=config.logging.train.log_every_n_steps,
        plugins=None,
        precision=config.training.precision,
        max_epochs=config.training.max_epochs,
        max_steps=config.training.max_steps,
        strategy=strategy,
        sync_batchnorm=False if strategy is None else True,
    )
    assert isinstance(trainer.logger, TensorBoardLogger)
    trainer.fit(model=module, ckpt_path=ckpt_path, datamodule=data_module)

if __name__ == '__main__':
    main()
