 python train.py \
    --gin_configs=configs/360.gin \
    --gin_bindings="Config.data_dir = '/home/gpu/LTY/zipnerf-pytorch/nerf__llff_data/bicycle/sparse'" \
    --gin_bindings="Config.exp_name = 'test'" \
      --gin_bindings="Config.factor = 4"


