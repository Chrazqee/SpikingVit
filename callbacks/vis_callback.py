import random
from enum import Enum, auto
from typing import Any, Optional, Union, List

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT

from data.utils.types import ObjDetOutput
from utils.ev_repr_to_img import ev_repr_to_img
from utils.evaluation.prophesee.visualize.vis_utils import draw_bboxes, LABELMAP_GEN1, LABELMAP_GEN4_SHORT


class DetectionVizEnum(Enum):
    EV_IMG = auto()
    LABEL_IMG_PROPH = auto()
    PRED_IMG_PROPH = auto()


class DetectionVizCallback(Callback):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.log_config = config.logging

        dataset_name = config.dataset.name
        if dataset_name == 'gen1':
            self.label_map = LABELMAP_GEN1
        elif dataset_name == 'gen4':
            self.label_map = LABELMAP_GEN4_SHORT
        else:
            raise NotImplementedError

        self._training_has_started = False
        self._selected_val_batches = False

        self.buffer_entries = DetectionVizEnum
        self._val_batch_indices = list()
        self._buffer = None
        self._reset_buffer()

    def _reset_buffer(self):
        self._buffer = {entry: [] for entry in self.buffer_entries}

    def add_to_buffer(self, key: Enum, value: Union[np.ndarray, torch.Tensor]):
        if isinstance(value, torch.Tensor):
            assert not value.requires_grad
            value = value.cpu()
        else:
            assert isinstance(value, np.ndarray)
        assert type(key) == self.buffer_entries
        assert key in self._buffer
        self._buffer[key].append(value)

    def get_from_buffer(self, key: Enum) -> List[torch.Tensor]:
        assert type(key) == self.buffer_entries
        return self._buffer[key]

    @rank_zero_only
    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int
    ) -> None:
        log_train_hd = self.log_config.train.high_dim
        if not log_train_hd.enable:
            return

        step = trainer.global_step
        assert log_train_hd.every_n_steps > 0
        if step % log_train_hd.every_n_steps != 0:
            return

        n_samples = log_train_hd.n_samples

        logger: Optional[TensorBoardLogger] = trainer.logger
        assert isinstance(logger, TensorBoardLogger)

        global_step = trainer.global_step

        self.on_train_batch_end_custom(
            logger=logger,
            outputs=outputs,
            batch=batch,
            log_n_samples=n_samples,
            global_step=global_step)

    @rank_zero_only
    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        log_val_hd = self.log_config.validation.high_dim
        log_freq_val_epochs = log_val_hd.every_n_epochs
        if not log_val_hd.enable:
            return
        if dataloader_idx > 0:
            raise NotImplementedError
        if not self._training_has_started:
            # PL has a short sanity check for validation. Hence, we have to make sure that one training run is done.
            return
        if not self._selected_val_batches:
            # We only want to add validation batch indices during the first true validation run.
            self._val_batch_indices.append(batch_idx)
            return
        assert len(self._val_batch_indices) > 0
        if batch_idx not in self._val_batch_indices:
            return
        if trainer.current_epoch % log_freq_val_epochs != 0:
            return

        self.on_validation_batch_end_custom(batch, outputs)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        log_val_hd = self.log_config.validation.high_dim
        log_n_samples = log_val_hd.n_samples
        log_freq_val_epochs = log_val_hd.every_n_epochs
        if len(self._val_batch_indices) == 0:
            return
        if not self._selected_val_batches:
            random.seed(0)
            num_samples = min(len(self._val_batch_indices), log_n_samples)
            # draw without replacement
            sampled_indices = random.sample(self._val_batch_indices, num_samples)
            self._val_batch_indices = sampled_indices
            self._selected_val_batches = True
            return
        if trainer.current_epoch % log_freq_val_epochs != 0:
            return

        logger: Optional[TensorBoardLogger] = trainer.logger
        assert isinstance(logger, TensorBoardLogger)
        global_step = trainer.global_step
        self.on_validation_epoch_end_custom(logger, global_step)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._reset_buffer()

    def on_train_batch_end_custom(self,
                                  logger: TensorBoardLogger,  # trainer.logger
                                  outputs: Any,
                                  batch: Any,
                                  log_n_samples: int,
                                  global_step: int) -> None:
        if outputs is None:
            # If we tried to skip the training step (not supported in DDP in PL, atm)
            return
        '''
        output = {
            ObjDetOutput.LABELS_PROPH: loaded_labels_proph[-1],
            ObjDetOutput.PRED_PROPH: yolox_preds_proph[-1],
            ObjDetOutput.EV_REPR: ev_repr_selector.get_event_representations_as_list(start_idx=-1)[0],
            ObjDetOutput.SKIP_VIZ: False,
        }
        '''
        ev_tensors = outputs[ObjDetOutput.EV_REPR]
        num_samples = len(ev_tensors)
        assert num_samples > 0
        log_n_samples = min(num_samples, log_n_samples)

        # merged_img = []
        # captions = []
        start_idx = num_samples - 1
        end_idx = start_idx - log_n_samples
        # for sample_idx in range(log_n_samples):
        key = 'train/predictions'
        for sample_idx in range(start_idx, end_idx, -1):
            ev_img = ev_repr_to_img(ev_tensors[sample_idx].cpu().numpy())

            predictions_proph = outputs[ObjDetOutput.PRED_PROPH][sample_idx]
            prediction_img = ev_img.copy()
            draw_bboxes(prediction_img, predictions_proph, labelmap=self.label_map)

            labels_proph = outputs[ObjDetOutput.LABELS_PROPH][sample_idx]
            label_img = ev_img.copy()
            draw_bboxes(label_img, labels_proph, labelmap=self.label_map)

            merged_img = torch.tensor(rearrange([prediction_img, label_img], 'pl H W C -> (pl H) W C', pl=2, C=3))
            logger.experiment.add_image(tag=key+f'_sample_{sample_idx}', img_tensor=merged_img, global_step=global_step, dataformats='HWC')

        # images = merged_img
        # if not isinstance(images, list):
        #     raise TypeError(f'Expected a list as "images", found {type(images)}')
        # n = len(images)
        # k, v = 'tag', captions
        # if len(v) != n:
        #     raise ValueError(f"Expected {n} items but only found {len(v)} for {k}")
        # kwarg_list = [{k: i} for i in v]
        # metrics = {
        #     key: [logger.experiment.add_image(img_tensor=img, dataformats='HWC', **kwarg) for img, kwarg in zip(images, kwarg_list)]}
        # logger.log_metrics(metrics, global_step)

        # if isinstance(merged_img[0], torch.Tensor):
        #     images = torch.tensor([img for img in merged_img])
        #     logger.experiment.add_images(tag=key, img_tensor=images, dataformats='NHWC', global_step=global_step)
        # else:
        #     images = np.asarray([img for img in merged_img])
        #     logger.experiment.add_images(tag=key, img_tensor=images, dataformats='NHWC', global_step=global_step)

    def on_validation_batch_end_custom(self, batch: Any, outputs: Any):
        if outputs[ObjDetOutput.SKIP_VIZ]:
            return
        ev_tensor = outputs[ObjDetOutput.EV_REPR]
        assert isinstance(ev_tensor, torch.Tensor)

        ev_img = ev_repr_to_img(ev_tensor.cpu().numpy())

        predictions_proph = outputs[ObjDetOutput.PRED_PROPH]
        prediction_img = ev_img.copy()
        draw_bboxes(prediction_img, predictions_proph, labelmap=self.label_map)
        self.add_to_buffer(DetectionVizEnum.PRED_IMG_PROPH, prediction_img)

        labels_proph = outputs[ObjDetOutput.LABELS_PROPH]
        label_img = ev_img.copy()
        draw_bboxes(label_img, labels_proph, labelmap=self.label_map)
        self.add_to_buffer(DetectionVizEnum.LABEL_IMG_PROPH, label_img)

    def on_validation_epoch_end_custom(self, logger: TensorBoardLogger, global_step):
        pred_imgs = self.get_from_buffer(DetectionVizEnum.PRED_IMG_PROPH)
        label_imgs = self.get_from_buffer(DetectionVizEnum.LABEL_IMG_PROPH)
        assert len(pred_imgs) == len(label_imgs)

        key = 'val/predictions'

        for idx, (pred_img, label_img) in enumerate(zip(pred_imgs, label_imgs)):
            merged_img = torch.tensor(rearrange([pred_img, label_img], 'pl H W C -> (pl H) W C', pl=2, C=3))  # 预测和 gt 合并在一块，上下合并
            logger.experiment.add_image(tag=key + f'_sample_{idx}', img_tensor=merged_img, global_step=global_step, dataformats='HWC')

        # images = merged_img
        # if not isinstance(images, list):
        #     raise TypeError(f'Expected a list as "images", found {type(images)}')
        # n = len(images)
        # k, v = 'tag', captions
        # if len(v) != n:
        #     raise ValueError(f"Expected {n} items but only found {len(v)} for {k}")
        # kwarg_list = [{k: i} for i in v]
        # if isinstance(merged_img[0], torch.Tensor):
        #     images = torch.tensor([img for img in merged_img])
        #     logger.experiment.add_images(tag=key, img_tensor=images, dataformats='NHWC')
        # else:
        #     images = np.asarray([img for img in merged_img])
        #     logger.experiment.add_images(tag=key, img_tensor=images, dataformats='NHWC')
        # metrics = {
        #     key: [logger.experiment.add_image(img_tensor=img, dataformats='HWC', **kwarg) for img, kwarg in
        #           zip(images, kwarg_list)]}
        # logger.log_metrics(metrics)
