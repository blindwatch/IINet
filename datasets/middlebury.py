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
from PIL import Image, ImageEnhance


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

class MiddleBuryDataset(Dataset):
    def __init__(self, mid_datapath, height, width, mode='train'):
        if mode == 'test':
            self.datapath = os.path.join(mid_datapath, 'MiddEval3', 'trainingH')
        elif mode == 'eval':
            self.datapath = os.path.join(mid_datapath, 'MiddEval3', 'trainingH')
        elif mode == 'train':
            self.datapath = os.path.join(mid_datapath, 'Middlebury2014')
        self.height = height
        self.width = width
        self.mode = mode
        self.process = get_transform()
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path()
        if self.mode == 'train':
            assert self.disp_filenames is not None
        self.augmentor = Augmentor(
            image_height=384,
            image_width=512,
            max_disp=192,
            scale_min=0.6,
            scale_max=1.0,
            seed=0,
        )
        self.rng = np.random.RandomState(0)

    def load_path(self):

        subfolders = os.listdir(self.datapath)
        if self.mode != 'train':
            left_images = [os.path.join(self.datapath, x, 'im0.png') for x in subfolders]
            right_images = [os.path.join(self.datapath, x, 'im1.png') for x in subfolders]
            if not self.mode == 'test':
                disp_images = [os.path.join(self.datapath,  x, 'disp0GT.pfm') for x in subfolders]
                return left_images, right_images, disp_images
            else:
                left_images = [os.path.join(self.datapath,'testing', x, 'im0.png') for x in subfolders]
                right_images = [os.path.join(self.datapath,'testing', x, 'im1.png') for x in subfolders]
                return left_images, right_images, None
        else:
            left_images = []
            right_images = []
            disp_images = []
            for x in subfolders:
                for s in ["E", "L", ""]:
                    left_images.append(os.path.join(self.datapath, str(x) + "/im0.png"))
                    right_images.append(os.path.join(self.datapath, str(x) + f"/im1{s}.png"))
                    disp_images.append(os.path.join(self.datapath, str(x) + f"/disp0.pfm"))
            return left_images,right_images,disp_images


    def load_image(self, filename):
        if self.mode != 'train':
            return np.array(Image.open(filename).convert('RGB'))
        else:
            img = np.array(Image.open(filename).convert('RGB'))
            return img[::2, ::2]

    def load_disp(self, filename):
        disparity, scale = pfm_imread(filename)
        # loaded disparity contains infs for no reading
        disparity[disparity == np.inf] = 0
        if self.mode != 'train':
            return np.ascontiguousarray(disparity)
        else:
            return np.ascontiguousarray(disparity[::2, ::2]) / 2

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):

        result = {}
        result['left'] = self.load_image(self.left_filenames[index])
        result['right'] = self.load_image(self.right_filenames[index])

        if self.disp_filenames:  # has disparity ground truth
            result['disp_pyr'] = self.load_disp(self.disp_filenames[index])

        if self.mode == 'train':
            h,w = result['left'].shape[:2]
            result['pos'] = _get_pos_fullres(800, w, h)
            result = random_crop(self.height, self.width, result)

            if self.rng.binomial(1, 0.2):
                result['left'], result['right'], result['disp_pyr'] = self.augmentor(
                    result['left'], result['right'], result['disp_pyr']
                )

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
            if 'disp_pyr' in result.keys():
                assert len(result['disp_pyr'].shape) == 2
                result['disp_pyr'] = np.lib.pad(result['disp_pyr'], ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            result['top_pad'] = top_pad
            result['right_pad'] = right_pad
            result["left_filename"] = self.left_filenames[index]
            result["right_filename"] = self.right_filenames[index]

        return result
