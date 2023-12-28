python val_or_test.py dataset=gen1 dataset.path=/home/chrazqee/datasets/gen1/ \
model=rnndet +encoder_model=encoder \
checkpoint= <your checkpoint path> \
hardware.gpus=1 +experiment/gen1='base.yaml' batch_size.eval=1 \
use_test_set=1 model.postprocess.confidence_threshold=0.001 \
dataset.downsample_by_factor_2=False  # set to False when using gen1 dataset!

