import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import os
from PIL import Image, ImageEnhance
from datasets.dataio import get_transform, read_all_lines, pfm_imread
import torch.nn.functional as F
from datasets.dataio import _get_pos_fullres, random_crop



class Augmentor:
    def __init__(
        self,
        scale_min=0.6,
        scale_max=1.0,
        seed=0,
    ):
        super().__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rng = np.random.RandomState(seed)

    def chromatic_augmentation(self, img, scale=1.0):
        random_brightness = np.random.uniform(0.8, 1.2) * scale
        random_contrast = np.random.uniform(0.8, 1.2)* scale
        random_gamma = np.random.uniform(0.8, 1.2)* scale

        img = Image.fromarray(img)

        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random_brightness)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random_contrast)

        gamma_map = [
            255 * 1.0 * pow(ele / 255.0, random_gamma) for ele in range(256)
        ] * 3
        img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

        img_ = np.array(img)

        return img_

    def __call__(self, left_img, right_img, left_disp):
        # 1. chromatic augmentation
        #scale = np.random.uniform(0.8, 1.2, 2)
        #left_img = self.chromatic_augmentation(left_img, scale[0])
        #right_img = self.chromatic_augmentation(right_img, scale[1])
        left_img = self.chromatic_augmentation(left_img)
        right_img = self.chromatic_augmentation(right_img)

        # 2. spatial augmentation
        # 2.1) rotate & vertical shift for right image
        if self.rng.binomial(1, 0.5):
            angle, pixel = 0.1, 2
            px = self.rng.uniform(-pixel, pixel)
            ag = self.rng.uniform(-angle, angle)
            image_center = (
                self.rng.uniform(0, right_img.shape[0]),
                self.rng.uniform(0, right_img.shape[1]),
            )
            rot_mat = cv2.getRotationMatrix2D(image_center, ag, 1.0)
            right_img = cv2.warpAffine(
                right_img, rot_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
            )
            trans_mat = np.float32([[1, 0, 0], [0, 1, px]])
            right_img = cv2.warpAffine(
                right_img, trans_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
            )

        return left_img, right_img, left_disp


class KITTIDataset(Dataset):
    def __init__(self, kitti_datapath, list_filename, height, width,mode='train'):
        self.datapath_15 = os.path.join(kitti_datapath, 'KITTI_2015')
        self.datapath_12 = os.path.join(kitti_datapath, 'KITTI_2012')
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.height = height
        self.width = width
        self.mode = mode
        self.augmentor = Augmentor(
            scale_min=0.6,
            scale_max=1.0,
            seed=0,
        )
        self.process = get_transform()
        self.rng = np.random.RandomState(0)
        if self.mode == 'train':
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        return np.array(Image.open(filename).convert('RGB'))

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):

        left_name = self.left_filenames[index].split('/')[1]
        if left_name.startswith('image'):
            self.datapath = self.datapath_15
        else:
            self.datapath = self.datapath_12
        result = {}
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))


        if self.disp_filenames:  # has disparity ground truth
            left_disp = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        else:
            left_disp = None

        if self.mode == 'train':

            #if self.rng.binomial(1, 0.2):
            #    left_img, right_img, left_disp = self.augmentor(
            #        left_img, right_img, left_disp
            #    )
            result['left'] = np.ascontiguousarray(left_img)
            result['right'] = np.ascontiguousarray(right_img)
            result['disp_pyr'] = np.ascontiguousarray(left_disp)
            result['pos'] = _get_pos_fullres(800, 1242, 375)
            result = random_crop(self.height, self.width, result, y_down=True, x_left=True)

            top_pad = 16
            right_pad = 16

            result['pos'] = np.lib.pad(result['pos'], ((0,0),(top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            result['left'] = np.lib.pad(result['left'], ((top_pad, 0), (0, right_pad),(0,0)), mode='constant', constant_values=0)
            result['right'] = np.lib.pad(result['right'], ((top_pad, 0), (0, right_pad),(0,0)), mode='constant', constant_values=0)
            result['disp_pyr'] = np.lib.pad(result['disp_pyr'], ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            #result = random_crop(self.height, self.width, result, y_down=False)
            result['left'] = self.process(result['left'])
            result['right'] = self.process(result['right'])

        else:

            result['left'] = np.ascontiguousarray(left_img)
            result['right'] = np.ascontiguousarray(right_img)
            result['pos'] = _get_pos_fullres(800, 1248, 384)
            # pad to size 1248x384
            h, w ,c = result['left'].shape
            top_pad = 384 - h
            right_pad = 1248 - w
            assert top_pad > 0 and right_pad > 0
            # pad images
            result['left'] = np.lib.pad(result['left'], ((top_pad, 0), (0, right_pad),(0,0)), mode='constant', constant_values=0)
            result['right'] = np.lib.pad(result['right'], ((top_pad, 0), (0, right_pad),(0,0)), mode='constant', constant_values=0)

            # pad disparity gt
            if left_disp is not None:
                assert len(left_disp.shape) == 2
                result['disp_pyr'] = np.lib.pad(left_disp, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            result['left'] = self.process(result['left'])
            result['right'] = self.process(result['right'])
            result['top_pad'] = top_pad
            result['right_pad'] = right_pad
            result["left_filename"] = self.left_filenames[index]
            result["right_filename"] = self.right_filenames[index]

        return result
