#!/usr/bin/env bash
python test_single.py --config_file configs/models/dot_model.yaml \
            --data_config configs/data/sceneflow.yaml \
            --load_weights_from_checkpoint sceneflow.pth