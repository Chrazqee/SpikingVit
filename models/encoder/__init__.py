# from omegaconf import DictConfig
#
# from models.encoder.spikingTransformerEncoder import build_transformer_encoder
#
#
# def build_encoder(encoder_cfg: DictConfig):
#     name = encoder_cfg.name
#     if name == 'spikingDETREncoder':
#         return build_transformer_encoder(encoder_cfg)  # 一个实例
#     else:
#         raise NotImplementedError
#
