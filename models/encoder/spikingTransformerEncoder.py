from typing import Dict

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from data.utils.types import SnnStates, FeatureMap
from .layers import (SSA,
                     MLP,
                     PatchMerging,
                     SN)


class EncoderBlocks(nn.Module):
    def __init__(self, dim: int, encoder_cfg: DictConfig):
        super().__init__()
        attention_cfg = encoder_cfg.attention
        assert isinstance(dim, int)
        assert isinstance(encoder_cfg, DictConfig)
        self.batchnorm1 = nn.BatchNorm2d(num_features=dim)
        self.ssa = SSA(dim, attention_cfg)
        self.batchnorm2 = nn.BatchNorm2d(num_features=dim)
        self.mlp = MLP(dim, dim * 2, dim)

    def forward(self, x):
        x = self.batchnorm1(x)
        x = x + self.ssa(x)
        x = self.batchnorm2(x)
        x = x + self.mlp(x)
        return x


# 🤔🤗 🤔🤗 🤔🤗 🤔🤗 🤔🤗 🤔🤗 🤔🤗 🤔🤗 🤔🤗 🤔🤗 🤔🤗 🤔🤗 🤔🤗
class Multi_scaleTransformerEncoder(nn.Module):
    def __init__(self, encoder_cfg: DictConfig):
        super().__init__()
        dim_multiplier_per_stage = encoder_cfg.get('dim_multiplier', None)
        num_blocks_per_stage = encoder_cfg.get('num_blocks', None)
        num_stages = len(num_blocks_per_stage)
        assert num_stages == 4
        assert num_stages == len(dim_multiplier_per_stage) or dim_multiplier_per_stage is not None
        assert num_stages == len(num_blocks_per_stage) or num_blocks_per_stage is not None

        self.num_stages = num_stages
        block_base = EncoderBlocks
        self.blocks = _get_clones(block_base, num_blocks_per_stage, encoder_cfg=encoder_cfg)

    def forward(self, x, prev_states):
        if prev_states is None:
            prev_states = [None] * self.num_stages
        assert len(prev_states) == self.num_stages

        states: SnnStates = []
        output: Dict[int, FeatureMap] = {}

        for stage_idx, layer in enumerate(self.blocks):
            patch_merging, encoder_blocks, sn = layer
            x = patch_merging(x)
            x = encoder_blocks(x)
            x, cur_state = sn(x, prev_states[stage_idx])
            states.append(cur_state)
            stage_number = stage_idx + 1
            output[stage_number] = x
        return output, states


def _get_clones(module, num_blocks_per_stage, encoder_cfg) -> nn.ModuleList:
    encoder_input_channels = encoder_cfg.get('encoder_input_channels', 64)
    assert isinstance(encoder_input_channels, int)
    dim_multiplier = encoder_cfg.get('dim_multiplier', None)
    assert dim_multiplier is not None
    model_list = nn.ModuleList()
    for dim_multi, num_blocks in zip(dim_multiplier, num_blocks_per_stage):

        channel_dim = dim_multi * encoder_input_channels
        model_list.append(nn.Sequential(PatchMerging(channel_dim // 2),
                                        nn.Sequential(*[module(channel_dim, encoder_cfg) for _ in range(num_blocks)]),
                                        SN(channel_dim)))
    encoder_model = model_list
    return encoder_model


def build_transformer_encoder(encoder_model_config: DictConfig) -> Multi_scaleTransformerEncoder:
    transformer_encoder = Multi_scaleTransformerEncoder(encoder_model_config.encoder)
    return transformer_encoder


if __name__ == "__main__":
    @hydra.main(config_path='/home/chrazqee/repository/Backbone_YOLOX_FPN_Head/conf', config_name='train',
                version_base='1.2')
    def main(config):
        """
        for validation!
        """
        print('------ Configuration ------')
        print(OmegaConf.to_yaml(config))
        print('---------------------------')
        encoder_model_config = config.encoder_model
        input_x = torch.randn((4, 32, 192, 320)).to("cuda:0")
        encoder = build_transformer_encoder(encoder_model_config).to("cuda:0")
        output = encoder(input_x, None)
        print(output)

    main()
    print("\n")
