#!/usr/bin/env bash
python test_spring.py --config_file configs/models/dot_model.yaml \
            --data_config configs/data/spring.yaml \
            --load_weights_from_checkpoint run/spring/spring192all_nenh_5e-5/models/model.pth.tar \
            --dataset spring \
            --expname test_spring192_nenh5e-5