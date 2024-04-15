#!/usr/bin/env bash
python train.py --config_file configs/models/dot_model.yaml \
            --data_config configs/data/spring.yaml \
            --load_weights_from_checkpoint spring.pth \
            --run_mode eval \
            --expname eval_spring \
            --summary_freq_eval 10