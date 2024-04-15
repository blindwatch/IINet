""" 
    Trains a DepthModel model. Uses an MVS dataset from datasets.

    - Outputs logs and checkpoints to opts.log_dir/opts.name
    - Supports mixed precision training by setting '--precision 16'

    We train with a batch_size of 16 with 16-bit precision on two A100s.

    Example command to train with two GPUs
        python train.py --name HERO_MODEL \
                    --log_dir logs \
                    --config_file configs/models/hero_model.yaml \
                    --data_config configs/data/scannet_default_train.yaml \
                    --gpus 2 \
                    --batch_size 16;
                    
"""


import options
import os
import random

import numpy as np
import torch
import torch.nn as nn

from datasets import build_data_loader
from modules.disp_model import DispModel
from modules.loss import build_criterion
from utils.checkpoint_saver import Saver, save_checkpoint
from utils.summary_logger import TensorboardSummary
from utils.foward_utils import train_one_epoch, evaluate
from utils.utils import adjust_learning_rate

torch.backends.cudnn.benchmark = True
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy("file_system")

def train(opts):
    # get device
    device = torch.device('cuda:0')

    # fix the seed for reproducibility
    seed = opts.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # build model
    model = DispModel(opts).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr, betas=(0.9, 0.999), weight_decay=opts.wd)

    # load checkpoint if provided
    prev_best = np.inf
    if opts.load_weights_from_checkpoint:
        checkpoint = torch.load(opts.load_weights_from_checkpoint)

        pretrained_dict = checkpoint['state_dict']
        missing, unexpected = model.load_state_dict(pretrained_dict, strict=False)
        # check missing and unexpected keys
        if len(missing) > 0:
            print("Missing keys: ", ','.join(missing))
            raise Exception("Missing keys.")
        unexpected_filtered = [k for k in unexpected if
                               'running_mean' not in k and 'running_var' not in k]  # skip bn params
        if len(unexpected_filtered) > 0:
            print("Unexpected keys: ", ','.join(unexpected_filtered))
            raise Exception("Unexpected keys.")
        print("Pre-trained model successfully loaded.")

        if opts.run_mode == 'train':
            if not opts.ft:
                opts.start_epoch = checkpoint['epoch'] + 1
                optimizer.load_state_dict(checkpoint['optimizer'])
                prev_best = checkpoint['best_pred']
                print("Pre-trained optimizer, lr scheduler and stats successfully loaded.")

    checkpoint_saver = Saver(opts)

    summary_writer = TensorboardSummary(checkpoint_saver.summary_dir)

    # dumping a copy of the config to the directory for easy(ier)
    # reproducibility.
    options.OptionsHandler.save_options_as_yaml(
                                    os.path.join(checkpoint_saver.directory, "config.yaml"),
                                    opts)

    # build dataloader
    data_loader_train, data_loader_val, _ = build_data_loader(opts)

    # build loss criterion
    criterion = build_criterion(opts)


    # eval
    if opts.run_mode == 'eval':
        print("Start evaluation")
        evaluate(model, criterion, data_loader_val, device, 60, opts, summary_writer)
        return

    # train
    print("Start training")
    for epoch in range(opts.start_epoch, opts.epochs):
        adjust_learning_rate(optimizer, epoch, opts.lr, opts.lrepochs)
        train_one_epoch(model, data_loader_train, optimizer, criterion, device, epoch, summary_writer,
                    opts)

        if epoch % opts.save_freq == 0:
            save_checkpoint(epoch, model, optimizer, prev_best, checkpoint_saver, False)

        # validate
        if epoch % opts.eval_freq == 0:
            evaluate(model, criterion, data_loader_val, device, epoch, opts, summary_writer)
        # save if best
            if prev_best > summary_writer.avgMeter.avg_data['epe'] and 5 > summary_writer.avgMeter.avg_data['px_error']:
                prev_best = summary_writer.avgMeter.avg_data['epe']
                save_checkpoint(epoch, model, optimizer, prev_best, checkpoint_saver, True)
        torch.cuda.empty_cache()

    evaluate(model, criterion, data_loader_val, device, epoch, opts, summary_writer)
    save_checkpoint(epoch, model, optimizer, prev_best, checkpoint_saver, False)

    return


if __name__ == '__main__':
    # get an instance of options and load it with config file(s) and cli args.
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options()
    option_handler.pretty_print_options()
    print("\n")
    opts = option_handler.options

    # if no GPUs are available for us then, use the 32 bit on CPU
    if opts.gpus == 0:
        print("Setting precision to 32 bits since --gpus is set to 0.")
        opts.precision = 32

    train(opts)
