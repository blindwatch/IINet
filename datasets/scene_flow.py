#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import os
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import torch.utils.data as data
from scipy.ndimage import zoom
from datasets.dataio import _get_pos_fullres


from datasets.dataio import read_all_lines, pfm_imread, get_transform, random_crop

class Augmentor:
    def __init__(
        self,
        image_height=384,
        image_width=512,
        max_disp=256,
        scale_min=0.6,
        scale_max=1.0,
        seed=0,
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.max_disp = max_disp
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rng = np.random.RandomState(seed)

    def chromatic_augmentation(self, img):
        random_brightness = np.random.uniform(0.8, 1.2)
        random_contrast = np.random.uniform(0.8, 1.2)
        random_gamma = np.random.uniform(0.8, 1.2)

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
        # 2.2) random resize
        '''
        resize_scale = self.rng.uniform(self.scale_min, self.scale_max)

        left_img = cv2.resize(
            left_img,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        right_img = cv2.resize(
            right_img,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_LINEAR,
        )

        disp_mask = (left_disp < float(self.max_disp / resize_scale)) & (left_disp > 0)
        disp_mask = disp_mask.astype("float32")
        disp_mask = cv2.resize(
            disp_mask,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_LINEAR,
        )

        left_disp = (
            cv2.resize(
                left_disp,
                None,
                fx=resize_scale,
                fy=resize_scale,
                interpolation=cv2.INTER_LINEAR,
            )
            * resize_scale
        )    

        # 2.3) random crop
        h, w, c = left_img.shape
        dx = w - self.image_width
        dy = h - self.image_height
        dy = self.rng.randint(min(0, dy), max(0, dy) + 1)
        dx = self.rng.randint(min(0, dx), max(0, dx) + 1)

        M = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
        left_img = cv2.warpAffine(
            left_img,
            M,
            (self.image_width, self.image_height),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )
        right_img = cv2.warpAffine(
            right_img,
            M,
            (self.image_width, self.image_height),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )
        left_disp = cv2.warpAffine(
            left_disp,
            M,
            (self.image_width, self.image_height),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )
        # 3. add random occlusion to right image
        if self.rng.binomial(1, 0.5):
            sx = int(self.rng.uniform(50, 100))
            sy = int(self.rng.uniform(50, 100))
            cx = int(self.rng.uniform(sx, right_img.shape[0] - sx))
            cy = int(self.rng.uniform(sy, right_img.shape[1] - sy))
            right_img[cx - sx : cx + sx, cy - sy : cy + sy] = np.mean(
                np.mean(right_img, 0), 0
            )[np.newaxis, np.newaxis]
        '''

        return left_img, right_img, left_disp


class SceneFlowDataset(data.Dataset):
    def __init__(self, datadir, list_filename, height, width,training):
        super(SceneFlowDataset, self).__init__()

        self.height = height
        self.width = width
        self.datadir = datadir
        self.training = training
        self.left_filenames, self.right_filenames, self.disp_filenames = self._load_path(list_filename)
        self.process = get_transform()
        self.augmentor = Augmentor(
            image_height=384,
            image_width=512,
            max_disp=192,
            scale_min=0.6,
            scale_max=1.0,
            seed=0,
        )
        self.rng = np.random.RandomState(0)

    def _load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        return left_images, right_images, disp_images

        self.rng = np.random.RandomState(seed)

    def _load_image(self, filename):
        return np.array(Image.open(filename).convert('RGB'))

    def _load_disp(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data


    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        result = {}

        result['pos'] = _get_pos_fullres(800, 960, 540)

        left_img = self._load_image(os.path.join(self.datadir, self.left_filenames[index]))
        right_img = self._load_image(os.path.join(self.datadir, self.right_filenames[index]))
        left_disp = self._load_disp(os.path.join(self.datadir, self.disp_filenames[index]))

        if self.training:
            right_disp = self._load_disp(os.path.join(self.datadir, self.disp_filenames[index].replace('left', 'right')))

            if self.rng.binomial(1, 0.5):
                left_img, right_img = np.fliplr(right_img), np.fliplr(left_img)
                left_disp, right_disp = np.fliplr(right_disp), np.fliplr(left_disp)
                left_disp[left_disp == np.inf] = 0

            if self.rng.binomial(1, 0.2):
                left_img, right_img, left_disp = self.augmentor(
                    left_img, right_img, left_disp
                )


        result['left'] = np.ascontiguousarray(left_img)
        result['right'] = np.ascontiguousarray(right_img)
        result['disp_pyr'] = np.ascontiguousarray(left_disp)

        result = random_crop(self.height, self.width, result)

        result['left'] = self.process(result['left'])
        result['right'] = self.process(result['right'])
        result['left_name'] = self.left_filenames[index]

        return result
