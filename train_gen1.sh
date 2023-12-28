# gen1
python train.py model=rnndet encoder_model=encoder dataset=gen1 +experiment/gen1="base.yaml" \
dataset.path=<your dataset path> dataset.downsample_by_factor_2=False hardware.gpus=[0] \
batch_size.train=24 batch_size.eval=24 hardware.num_workers.train=2 hardware.num_workers.eval=2 \
validation.val_check_interval=10000 validation.check_val_every_n_epoch=null training.max_steps=300000 \
dataset.sequence_length=5 training.learning_rate=0.0002 encoder_model.encoder.num_blocks=[2,2,2,2]
