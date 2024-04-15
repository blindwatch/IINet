import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import os
from datasets.dataio import get_transform, read_all_lines, pfm_imread
import torch.nn.functional as F
from datasets.dataio import _get_pos_fullres, random_crop


class Eth3dDataset(Dataset):
    def __init__(self, eth_datapath, scene_filename, height, width,mode='train'):
        self.datapath = eth_datapath
        self.height = height
        self.width = width
        self.mode = mode
        self.process = get_transform()
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(scene_filename)
        if self.mode == 'train':
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        subfolders = read_all_lines(list_filename)

        if not self.mode == 'test':
            left_images = [os.path.join(self.datapath, 'train', x, 'im0.png') for x in subfolders]
            right_images = [os.path.join(self.datapath, 'train', x, 'im1.png') for x in subfolders]
            disp_images = [os.path.join(self.datapath, 'train', x, 'disp0GT.pfm') for x in subfolders]
            return left_images, right_images, disp_images
        else:
            left_images = [os.path.join(self.datapath,'test', x, 'im0.png') for x in subfolders]
            right_images = [os.path.join(self.datapath,'test', x, 'im1.png') for x in subfolders]
            return left_images, right_images, None


    def load_image(self, filename):
        return np.array(Image.open(filename).convert('RGB'))

    def load_disp(self, filename):
        disparity,scale = pfm_imread(filename)
        # loaded disparity contains infs for no reading
        disparity[disparity == np.inf] = 0
        return np.ascontiguousarray(disparity)

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):

        result = {}
        result['left'] = self.load_image(self.left_filenames[index])
        result['right'] = self.load_image(self.right_filenames[index])

        if self.disp_filenames:  # has disparity ground truth
            result['disp_pyr'] = self.load_disp(self.disp_filenames[index])

        if self.mode=='train':
            h, w = result['left'].shape[:2]
            result['pos'] = _get_pos_fullres(800, w, h)
            result = random_crop(self.height, self.width, result)

            top_pad = 16
            right_pad = 16

            result['pos'] = np.lib.pad(result['pos'], ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                       constant_values=0)
            result['left'] = np.lib.pad(result['left'], ((top_pad, 0), (0, right_pad), (0, 0)), mode='constant',
                                        constant_values=0)
            result['right'] = np.lib.pad(result['right'], ((top_pad, 0), (0, right_pad), (0, 0)), mode='constant',
                                         constant_values=0)
            result['disp_pyr'] = np.lib.pad(result['disp_pyr'], ((top_pad, 0), (0, right_pad)), mode='constant',
                                            constant_values=-1)

            result['left'] = self.process(result['left'])
            result['right'] = self.process(result['right'])

        else:
            h, w = result['left'].shape[:2]
            result['pos'] = _get_pos_fullres(800, w, h)
            result['left'] = self.process(result['left'])
            result['right'] = self.process(result['right'])

            top_pad = 32 - (h % 32)
            right_pad = 32 - (w % 32)
            # pad images
            result['left'] = F.pad(result['left'], (0, right_pad, top_pad, 0))
            result['right'] = F.pad(result['right'], (0, right_pad, top_pad, 0))
            result['pos'] = np.lib.pad(result['pos'], ((0,0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            # pad disparity gt
            if 'disp_pyr' in result:
                assert len(result['disp_pyr'].shape) == 2
                result['disp_pyr'] = np.lib.pad(result['disp_pyr'], ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            result['top_pad'] = top_pad
            result['right_pad'] = right_pad
            result["left_filename"] = self.left_filenames[index]
            result["right_filename"] = self.right_filenames[index]

        return result
