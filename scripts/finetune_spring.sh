#!/usr/bin/env bash
python train.py --config_file configs/models/dot_model.yaml \
            --data_config configs/data/spring.yaml \
            --load_weights_from_checkpoint run/sceneflow/msr_wocaoc_0.2enh_longer_384/models/epoch_104_model.pth.tar \
            --ft \
            --expname spring192all_nenh_5e-5 \
            --summary_freq 50