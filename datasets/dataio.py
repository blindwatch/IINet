import numpy as np
import re
import h5py
import png
import torchvision.transforms as transforms
import random


def _get_pos_fullres(fx, w, h):
    x_range = (np.linspace(0, w - 1, w) + 0.5 - w // 2) / fx
    y_range = (np.linspace(0, h - 1, h) + 0.5 - h // 2) / fx
    x, y = np.meshgrid(x_range, y_range)
    z = np.ones_like(x)
    pos_grid = np.stack([x, y, z], axis=0).astype(np.float32)
    return pos_grid

def random_crop(min_crop_height, min_crop_width, input_data, y_down=False, x_left=False):
    """
    Crop center part of the input with a random width and height.

    :param min_crop_height: min height of the crop, int
    :param min_crop_width: min width of the crop, int
    :param input_data: input data, dictionary
    :param split: train/validation split, string
    :return: updated input data, dictionary
    """

    height, width = input_data['left'].shape[:2]

    crop_height = min_crop_height
    crop_width = min_crop_width

    crop_y = height - crop_height - 1
    if crop_y < 0:
        y1 = 0
    elif y_down == False or random.randint(0, 10) >= int(8):
        y1 = random.randint(0, crop_y)
    else:
        y1 = random.randint(int(0.2 * height), crop_y)
    y2 = y1 + crop_height

    crop_x = width - crop_width - 1
    if crop_x < 0:
        x1 = 0
    elif x_left == False or random.randint(0, 10) >= int(3):
        x1 = random.randint(0, crop_x)
    #elif random.randint(0, 10) >= int(7):
    #    x1 = random.randint(0, int(0.05 * width))
    #else:
    #    x1 = random.randint(crop_x - int(0.05 * width), crop_x)
    else:
     x1 = random.randint(0, int(0.05 * width))
    x2 = x1 + crop_width

    # x1, y1, x2, y2 = get_random_crop_coords(height, width, crop_height, crop_width)

    input_data['left'] = input_data['left'][y1:y2, x1:x2]
    input_data['right'] = input_data['right'][y1:y2, x1:x2]
    input_data['disp_pyr'] = input_data['disp_pyr'][y1:y2, x1:x2]
    input_data['pos'] = input_data['pos'][:, y1:y2, x1:x2]

    return input_data



def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])




# read all lines in a file
def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines


# read an .pfm file into numpy array, used to load SceneFlow disparity files
def pfm_imread(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def readPfmFile(filepath):
    """
    adapted from https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
    """
    file = open(filepath, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data #, scale

def readNpyFlow(filepath):
    """read numpy array from file.
    filepath: file to read from
    returns: numpy array
    """
    return np.load(filepath)


def writeNpyFile(arr, filepath):
    """write numpy array to file.
    arr: numpy array to write
    filepath: file to write to
    """
    np.save(filepath, arr)

def readDispFile(filepath):
    """read disparity (or disparity change) from file. The resulting numpy array has shape height x width.
    For positions where there is no groundtruth available, the value is set to np.nan.
    Supports png (KITTI), npy (numpy) and pfm (FlyingThings3D) file format.
    filepath: path to the flow file
    returns: disparity with shape height x width
    """
    if filepath.endswith(".png"):
        return readPngDisp(filepath)
    elif filepath.endswith(".npy"):
        return readNpyFlow(filepath)
    elif filepath.endswith(".pfm"):
        return readPfmDisp(filepath)
    elif filepath.endswith(".dsp5"):
        return readDsp5Disp(filepath)
    else:
        raise ValueError(f"readDispFile: Unknown file format for {filepath}")


def readPngDisp(filepath):
    """read disparity from file stored in png file format as used in the KITTI 12 (Geiger et al., 2012) and KITTI 15 (Menze et al., 2015) dataset.
    filepath: path to file where to read from
    returns: disparity as a numpy array with shape height x width. Invalid values are represented as np.nan
    """
    # adapted from https://github.com/liruoteng/OpticalFlowToolkit
    image_object = png.Reader(filename=filepath)
    image_direct = image_object.asDirect()
    image_data = list(image_direct[2])
    (w, h) = image_direct[3]['size']
    channel = len(image_data[0]) // w
    if channel != 1:
        raise IOError("read png disp: assumed channels to be 1!")
    disp = np.zeros((h, w), dtype=np.float64)
    for i in range(len(image_data)):
        disp[i, :] = image_data[i][:]
    disp[disp == 0] = np.nan
    return disp[:, :] / 256.0


def readPfmDisp(filepath):
    """read disparity or disparity change from file stored in pfm file format as used in the FlyingThings3D (Mayer et al., 2016) dataset.
    filepath: path to file where to read from
    returns: disparity as a numpy array with shape height x width. Invalid values are represented as np.nan
    """
    disp = readPfmFile(filepath)
    if len(disp.shape) != 2:
        raise IOError(f"read pfm disp: PFM file has wrong shape (assumed to be w x h): {disp.shape}")
    return disp


def writePngDisp(disp, filepath):
    """write disparity to png file format as used in the KITTI 12 (Geiger et al., 2012) and KITTI 15 (Menze et al., 2015) dataset.
    disp: disparity in shape height x width, invalid values should be represented as np.nan
    filepath: path to file where to write to
    """
    disp = 256 * disp
    width = disp.shape[1]
    height = disp.shape[0]
    disp = np.clip(disp, 0, 2 ** 16 - 1)
    disp = np.nan_to_num(disp).astype(np.uint16)
    disp = np.reshape(disp, (-1, width))
    with open(filepath, "wb") as f:
        writer = png.Writer(width=width, height=height, bitdepth=16, greyscale=True)
        writer.write(f, disp)


def writeDsp5File(disp, filename):
    with h5py.File(filename, "w") as f:
        f.create_dataset("disparity", data=disp, compression="gzip", compression_opts=5)


def readDsp5Disp(filename):
    with h5py.File(filename, "r") as f:
        if "disparity" not in f.keys():
            raise IOError(f"File {filename} does not have a 'disparity' key. Is this a valid dsp5 file?")
        return f["disparity"][()]


def writeDispFile(disp, filepath):
    """write disparity to file. Supports png (KITTI) and npy (numpy) file format.
    disp: disparity with shape height x width. Invalid values should be represented as np.nan
    filepath: file path where to write the flow
    """
    if not filepath:
        raise ValueError("writeDispFile: empty filepath")

    if len(disp.shape) != 2:
        raise IOError(f"writeDispFile {filepath}: expected shape height x width but received {disp.shape}")

    if disp.shape[0] > disp.shape[1]:
        print(
            f"writeDispFile {filepath}: Warning: Are you writing an upright image? Expected shape height x width, got {disp.shape}")

    if filepath.endswith(".png"):
        writePngDisp(disp, filepath)
    elif filepath.endswith(".npy"):
        writeNpyFile(disp, filepath)
    elif filepath.endswith(".dsp5"):
        writeDsp5File(disp, filepath)
