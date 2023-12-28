from typing import Tuple, Optional, List

from omegaconf import DictConfig, ListConfig
from torch import nn, Tensor

from data.utils.types import EncoderFeatures, SnnStates
from models.encoder.layers.temporal_extension import TemporalExtension
from models.encoder.spikingTransformerEncoder import build_transformer_encoder
from utils.timers import CudaTimer


class SpikingVit(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        assert isinstance(config, DictConfig)
        encoder_model_cfg = config.encoder_model
        assert isinstance(encoder_model_cfg, DictConfig)

        num_blocks = encoder_model_cfg.encoder.num_blocks
        assert isinstance(num_blocks, ListConfig)
        self.num_blocks = sum(num_blocks)

        # down_sampler before encoder
        downsample_cfg = encoder_model_cfg.downsample
        self.down_sampler = TemporalExtension(downsample_cfg)

        # encoder model instance
        self.mdl_encoder = build_transformer_encoder(encoder_model_cfg)

    def forward_encoder(self, x: Tensor, previous_states: Optional[SnnStates] = None) -> Tuple[EncoderFeatures, SnnStates]:
        with CudaTimer(device=x.device, timer_name='Encoder'):
            x = self.down_sampler(x)
            encoder_features, states = self.mdl_encoder(x, previous_states)
        return encoder_features, states
