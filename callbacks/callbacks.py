from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint

from callbacks.vis_callback import DetectionVizCallback


def get_ckpt_callback() -> ModelCheckpoint:
    prefix = 'val'
    metric = 'AP'
    mode = 'max'
    ckpt_callback_monitor = prefix + '/' + metric
    filename_monitor_str = prefix + '_' + metric
    ckpt_filename = 'epoch-{epoch:03d}-step-{step}-' + filename_monitor_str + '-{' + ckpt_callback_monitor + ':.2f}'
    ckpt_callback = ModelCheckpoint(
        monitor=ckpt_callback_monitor,
        filename=ckpt_filename,
        auto_insert_metric_name=False,  # because backslash would create a directory
        save_top_k=1,
        mode=mode,
        every_n_epochs=1,
        save_last=True,
        verbose=True)
    ckpt_callback.CHECKPOINT_NAME_LAST = 'last_epoch-{epoch:03d}-step-{step}'
    return ckpt_callback

def get_viz_callback(config: DictConfig) -> Callback:
    return DetectionVizCallback(config=config)
