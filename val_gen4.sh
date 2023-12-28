python val_or_test.py dataset=gen4 dataset.path=/home/chrazqee/datasets/gen4/ \
model=rnndet +encoder_model=encoder \
checkpoint= <your checkpoint path> \
use_test_set=1 hardware.gpus=0 +experiment/gen4='base.yaml' \
batch_size.eval=10 model.postprocess.confidence_threshold=0.001 \
dataset.downsample_by_factor_2=True  # set to False when using gen1 dataset!

