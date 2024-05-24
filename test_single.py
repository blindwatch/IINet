
from __future__ import print_function, division
import options
import skimage
import numpy as np
from modules.disp_model import DispModel
from datasets.dataio import writeDispFile

import torch
from utils.utils import tensor2numpy, savepreprocess
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import os

# cudnn.benchmark = True
pic1 = 'test/left.png'
pic2 = 'test/right.png'

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

@torch.no_grad()
def test(model: torch.nn.Module,
        device: torch.device):

    save_path = os.path.join('test')
    os.makedirs(save_path, exist_ok=True)
    model.eval()

    left_pic = np.ascontiguousarray(np.array(Image.open(pic1).convert('RGB')))
    right_pic = np.ascontiguousarray(np.array(Image.open(pic2).convert('RGB')))

    h, w, c = left_pic.shape
    top_pad = (32 - (h % 32)) % 32
    right_pad = (32 - (w % 32)) % 32

    if not (top_pad == 32 and right_pad == 32):
        left_pic = np.lib.pad(left_pic, ((top_pad, 0), (0, right_pad), (0, 0)), mode='constant',
                                    constant_values=0)
        right_pic = np.lib.pad(right_pic, ((top_pad, 0), (0, right_pad), (0, 0)), mode='constant',
                                     constant_values=0)

    left_pic = transform(left_pic)[None, ...]
    right_pic = transform(right_pic)[None, ...]

    inputs = {
        'left': left_pic.to(device),
        'right': right_pic.to(device),
    }

    # forward pass
    outputs_left = model(inputs)

    if right_pad != 0:
        disp_pred_np = tensor2numpy((16 * outputs_left['disp_pred_s0']).squeeze(1))[:, top_pad:, :-right_pad] #1,h,w
    else:
        disp_pred_np = tensor2numpy((16 * outputs_left['disp_pred_s0']).squeeze(1))[:, top_pad:, :] #1,h,w

    disp_show = os.path.join(save_path, 'color_disp.png')
    disp_uint = os.path.join(save_path, 'disp_16bit.png')
    plt.imsave(disp_show, savepreprocess(disp_pred_np) / 255.)
    skimage.io.imsave(disp_uint, np.round(disp_pred_np[0] * 256).astype(np.uint16))

    print("image saved")
    return

if __name__ == '__main__':
    # get an instance of options and load it with config file(s) and cli args.
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options()
    opts = option_handler.options

    device = torch.device('cuda')
    model = DispModel(opts).to(device)

    if not os.path.isfile(opts.load_weights_from_checkpoint):
        raise RuntimeError(f"=> no checkpoint found at '{opts.load_weights_from_checkpoint}'")
    checkpoint = torch.load(opts.load_weights_from_checkpoint)

    pretrained_dict = checkpoint['state_dict']
    model.load_state_dict(pretrained_dict, strict=True)

    test(model, device)


