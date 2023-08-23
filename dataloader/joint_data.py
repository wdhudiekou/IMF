import pathlib

import cv2
import numpy as np
import kornia.utils
import torch.utils.data


class JointTrainData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """
    def __init__(self, ir_folder: pathlib.Path, it_folder: pathlib.Path, vi_folder: pathlib.Path, ir_map: pathlib.Path, vi_map: pathlib.Path, crop=lambda x: x):
        super(JointTrainData, self).__init__()
        self.crop = crop
        # gain infrared and visible images list
        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.it_list = [x for x in sorted(it_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

        self.ir_map_list = [x for x in sorted(ir_map.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_map_list = [x for x in sorted(vi_map.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        it_path = self.it_list[index]
        vi_path = self.vi_list[index]

        ir_map_path = self.ir_map_list[index]
        vi_map_path = self.vi_map_list[index]

        # assert ir_path.name == vi_path.name, f"Mismatch ir:{ir_path.name} vi:{vi_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        it = self.imread(path=it_path, flags=cv2.IMREAD_GRAYSCALE)
        vi = self.imread(path=vi_path, flags=cv2.IMREAD_GRAYSCALE)

        ir_map = self.imread(path=ir_map_path, flags=cv2.IMREAD_GRAYSCALE)
        vi_map = self.imread(path=vi_map_path, flags=cv2.IMREAD_GRAYSCALE)

        return (ir, it, vi, ir_map, vi_map), (str(ir_path), str(it_path), str(vi_path))  # fake

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts


class JointTestData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    def __init__(self, ir_folder: pathlib.Path, it_folder: pathlib.Path, vi_folder: pathlib.Path, disp_folder: pathlib.Path):
        super(JointTestData, self).__init__()
        # gain infrared and visible images list
        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.it_list = [x for x in sorted(it_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.disp_list = [x for x in sorted(disp_folder.glob('*')) if x.suffix in ['.npy']]

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        it_path = self.it_list[index]
        vi_path = self.vi_list[index]
        disp_path = self.disp_list[index]

        assert ir_path.name == vi_path.name, f"Mismatch ir:{ir_path.name} vi:{vi_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path)
        it = self.imread(path=it_path)
        vi = self.imread(path=vi_path)
        disp = torch.from_numpy(np.load(disp_path))

        return (ir, it, vi, disp), (str(ir_path), str(it_path), str(vi_path))

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts

class JointRealData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    def __init__(self, ir_folder: pathlib.Path, it_folder: pathlib.Path, vi_folder: pathlib.Path):
        super(JointRealData, self).__init__()
        # gain infrared and visible images list
        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.it_list = [x for x in sorted(it_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        it_path = self.it_list[index]
        vi_path = self.vi_list[index]

        assert ir_path.name == vi_path.name, f"Mismatch ir:{ir_path.name} vi:{vi_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path)
        it = self.imread(path=it_path)
        vi = self.imread(path=vi_path)

        return (ir, it, vi), (str(ir_path), str(it_path), str(vi_path))

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts
