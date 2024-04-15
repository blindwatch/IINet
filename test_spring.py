
from __future__ import print_function, division
import options
import skimage
from utils.checkpoint_saver import Saver
import numpy as np
from datasets import build_data_loader
from modules.disp_model import DispModel
from datasets.dataio import writeDispFile

import torch
from dataclasses import dataclass
import torch.utils.data as data
from utils.utils import tensor2numpy, savepreprocess
import progressbar
import matplotlib.pyplot as plt
import os
import cv2
# cudnn.benchmark = True

@torch.no_grad()
def test(model: torch.nn.Module,
             data_loader: data.DataLoader,
             device: torch.device,
             opts: dataclass):

    pwidgets = [progressbar.Percentage(), " ", progressbar.Counter(format='%(value)02d/%(max_value)d'), " ",
                progressbar.Bar(), " ",
                progressbar.Timer(), ",", progressbar.ETA(), ",",
                ]
    pbar = progressbar.ProgressBar(widgets=pwidgets,
                                   max_value=len(data_loader),
                                   prefix="Test:").start()

    _ = Saver(opts)
    save_path = os.path.join(opts.logdir, opts.dataset, opts.expname, 'eval_pics')
    disp_root = os.path.join(opts.logdir, opts.dataset, opts.expname, 'spring')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(disp_root, exist_ok=True)
    model.eval()

    for batch_idx, data in enumerate(data_loader):
        # forward pass

        inputs_left = {
            'left': data['left'].to(device, non_blocking=True),
            'right': data['right'].to(device, non_blocking=True),
        }

        # forward pass
        outputs_left = model(inputs_left, False)

        inputs_right = {
            'left': data['right_fl'].to(device, non_blocking=True),
            'right': data['left_fl'].to(device, non_blocking=True),
        }
        outputs_right = model(inputs_right, False)

        left_disp_pred_np = tensor2numpy((16 * outputs_left['disp_pred_s0']).squeeze(1)) #b,h,w
        right_disp_pred_np = tensor2numpy((16 * outputs_right['disp_pred_s0']).squeeze(1))  # b,h,w

        top_pad_np = tensor2numpy(data["top_pad"])
        right_pad_np = tensor2numpy(data["right_pad"])
        left_filenames = data["left_filename"]

        for disp_estl, disp_estr, top_pad, right_pad, fn in zip(left_disp_pred_np, right_disp_pred_np, top_pad_np, right_pad_np, left_filenames):
            assert len(disp_estl.shape) == 2
            left_split_name = fn.split('/')
            left_split_name.pop(0)
            left_name = ('/').join(left_split_name)
            left_name = left_name.replace('.png', '.dsp5').replace('frame', 'disp1')
            right_name = left_name.replace('left', 'right')
            if right_pad != 0:
                disp_estl = np.array(disp_estl[top_pad:, :-right_pad], dtype=np.float32)
                disp_estr = np.array(np.fliplr(disp_estr[top_pad:, :-right_pad]), dtype=np.float32)
            else:
                disp_estl = np.array(disp_estl[top_pad:, :], dtype=np.float32)
                disp_estr = np.array(np.fliplr(disp_estr[top_pad:, :]), dtype=np.float32)

            left_submit = os.path.join(disp_root, left_name)
            os.makedirs(os.path.dirname(left_submit),  exist_ok=True)
            right_submit = os.path.join(disp_root, right_name)
            os.makedirs(os.path.dirname(right_submit), exist_ok=True)
            left_show = os.path.join(save_path, left_name.replace('.dsp5', '.png'))
            os.makedirs(os.path.dirname(left_show), exist_ok=True)
            right_show = os.path.join(save_path, right_name.replace('.dsp5', '.png'))
            os.makedirs(os.path.dirname(right_show), exist_ok=True)
            writeDispFile(disp_estl, left_submit)
            writeDispFile(disp_estr, right_submit)
            plt.imsave(left_show, savepreprocess(disp_estl[None,...]) / 255.)
            plt.imsave(right_show, savepreprocess(disp_estr[None, ...]) / 255.)
            print("saving to", left_show, disp_estr.shape)
            # cv2.imwrite(fn, cv2.applyColorMap(cv2.convertScaleAbs(disp_est_uint, alpha=0.01),cv2.COLORMAP_JET))

        pbar.update(batch_idx)

    pbar.finish()
    return

if __name__ == '__main__':
    # get an instance of options and load it with config file(s) and cli args.
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options()
    option_handler.pretty_print_options()
    print("\n")
    opts = option_handler.options

    device = torch.device('cuda:0')
    model = DispModel(opts).to(device)
    _,_, TestLoader = build_data_loader(opts)

    if opts.load_weights_from_checkpoint:
        if not os.path.isfile(opts.load_weights_from_checkpoint):
            raise RuntimeError(f"=> no checkpoint found at '{opts.load_weights_from_checkpoint}'")
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
    else:
        raise Exception("no weight is provided.")

    test(model, TestLoader, device, opts)


