# gen4
python train.py model=rnndet encoder_model=encoder dataset=gen4 +experiment/gen4="base.yaml" \
dataset.path=<your dataset path> dataset.downsample_by_factor_2=True hardware.gpus=[0] \
batch_size.train=10 batch_size.eval=10 hardware.num_workers.train=4 hardware.num_workers.eval=4 \
validation.val_check_interval=10000 validation.check_val_every_n_epoch=null training.max_steps=500000 \
dataset.sequence_length=1 training.learning_rate=0.0003 encoder_model.encoder.num_blocks=[2,2,2,2] \
model.backbone.embed_dim=64

