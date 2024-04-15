#!/usr/bin/env bash
python train.py --config_file configs/models/dot_model.yaml \
            --data_config configs/data/sceneflow.yaml \
            --load_weights_from_checkpoint sceneflow.pth \
            --save_eval \
            --run_mode eval \
            --expname eval_sceneflow \
            --summary_freq_eval 10
