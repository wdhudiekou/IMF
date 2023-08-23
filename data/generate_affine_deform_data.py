import pathlib
import os
import cv2
import kornia
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
from kornia.filters.kernels import get_gaussian_kernel2d


class AffineTransform(nn.Module):
    """
    Add random affine transforms to a tensor image.
    Most functions are obtained from Kornia, difference:
    - gain the disp grid
    - no p and same_on_batch
    """
    def __init__(self, degrees=5, translate=0.1, scale=1.0, shear=None):
        super(AffineTransform, self).__init__()
        self.trs = kornia.augmentation.RandomAffine(degrees, (translate, translate), (scale, scale), shear, return_transform=True, p=1)

    def forward(self, input):
        # image shape
        batch_size, _, height, weight = input.shape
        # affine transform
        warped, affine_param = self.trs(input)  # [batch_size, 3, 3]
        affine_theta = self.param_to_theta(affine_param, weight, height)  # [batch_size, 2, 3]
        # base + disp = grid -> disp = grid - base
        base = kornia.utils.create_meshgrid(height, weight, device=input.device).to(input.dtype)
        grid = F.affine_grid(affine_theta, size=input.size(), align_corners=False)  # [batch_size, height, weight, 2]
        disp = grid - base
        return warped, -disp

    @staticmethod
    def param_to_theta(param, weight, height):
        """
        Convert affine transform matrix to theta in F.affine_grid
        :param param: affine transform matrix [batch_size, 3, 3]
        :param weight: image weight
        :param height: image height
        :return: theta in F.affine_grid [batch_size, 2, 3]
        """

        theta = torch.zeros(size=(param.shape[0], 2, 3)).to(param.device)  # [batch_size, 2, 3]

        theta[:, 0, 0] = param[:, 0, 0]
        theta[:, 0, 1] = param[:, 0, 1] * height / weight
        theta[:, 0, 2] = param[:, 0, 2] * 2 / weight + param[:, 0, 0] + param[:, 0, 1] - 1
        theta[:, 1, 0] = param[:, 1, 0] * weight / height
        theta[:, 1, 1] = param[:, 1, 1]
        theta[:, 1, 2] = param[:, 1, 2] * 2 / height + param[:, 1, 0] + param[:, 1, 1] - 1

        return theta

class ElasticTransform(nn.Module):
    """
    Add random elastic transforms to a tensor image.
    Most functions are obtained from Kornia, difference:
    - gain the disp grid
    - no p and same_on_batch
    """

    def __init__(self, kernel_size: int = 63, sigma: float = 32, alpha: Tuple[float, float]= (1.0, 1.0), align_corners: bool = False, mode: str = "bilinear"):
        super(ElasticTransform, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.alpha = alpha
        self.align_corners = align_corners
        self.mode  = mode

    def forward(self, input):
        # generate noise
        batch_size, _, height, weight = input.shape
        noise = torch.rand(batch_size, 2, height, weight) * 2 - 1  # torch.Size([16, 2, 256, 320])
        # elastic transform
        warped, disp = self.elastic_transform2d(input, noise)
        return warped, disp

    def elastic_transform2d(self, image: torch.Tensor, noise: torch.Tensor):
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input image is not torch.Tensor. Got {type(image)}")

        if not isinstance(noise, torch.Tensor):
            raise TypeError(f"Input noise is not torch.Tensor. Got {type(noise)}")

        if not len(image.shape) == 4:
            raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {image.shape}")

        if not len(noise.shape) == 4 or noise.shape[1] != 2:
            raise ValueError(f"Invalid noise shape, we expect Bx2xHxW. Got: {noise.shape}")

        # unpack hyper parameters
        kernel_size = self.kernel_size
        sigma = self.sigma
        alpha = self.alpha
        align_corners = self.align_corners
        mode = self.mode
        device = image.device

        # Get Gaussian kernel for 'y' and 'x' displacement
        kernel_x: torch.Tensor = get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))[None]
        kernel_y: torch.Tensor = get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))[None]

        # Convolve over a random displacement matrix and scale them with 'alpha'
        disp_x: torch.Tensor = noise[:, :1].to(device)
        disp_y: torch.Tensor = noise[:, 1:].to(device)

        disp_x = kornia.filters.filter2d(disp_x, kernel=kernel_y, border_type="constant") * alpha[0]
        disp_y = kornia.filters.filter2d(disp_y, kernel=kernel_x, border_type="constant") * alpha[0]

        # stack and normalize displacement
        disp = torch.cat([disp_x, disp_y], dim=1).permute(0, 2, 3, 1)

        # Warp image based on displacement matrix
        b, c, h, w = image.shape
        grid = kornia.utils.create_meshgrid(h, w, device=image.device).to(image.dtype)
        warped = F.grid_sample(image, (grid + disp).clamp(-1, 1), align_corners=align_corners, mode=mode)

        return warped, disp

def imread(path, flags=cv2.IMREAD_GRAYSCALE):
    im_cv = cv2.imread(path, flags)
    assert im_cv is not None, f"Image {str(path)} is invalid."
    im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
    return im_ts

def imsave(im_s, dst):
    im_ts = im_s.squeeze().cpu()
    im_cv = kornia.utils.tensor_to_image(im_ts) * 255.
    cv2.imwrite(dst, im_cv)

def imfuse(img1, img2):
    img1 = img1.squeeze().cpu()
    img2 = img2.squeeze().cpu()
    im_cv_1 = kornia.utils.tensor_to_image(img1) * 255.
    im_cv_2 = kornia.utils.tensor_to_image(img2) * 255.

    img_fuse = np.dstack((im_cv_1, im_cv_2, im_cv_1))
    return img_fuse

def get_random_affine_coeff():
    # TODO: define transformation coefficent
    degree = random.uniform(0, 10) # light_defo: (0, 0); middle_defo: (0, 5); heavy_defo: (0, 10)
    translate = random.uniform(0.02, 0.05) # light_defo: (0.0, 0.01); middle_defo: (0.0, 0.02); heavy_defo: (0.02, 0.05)
    scale = random.uniform(1.0, 1.0)
    shear = None
    return degree, translate, scale, shear

def get_random_deformable_coeff():
    kernel = random.randrange(63, 101, 2)
    sigma = random.randint(16, 24) # light_defo: (32, 32); middle_defo: (16, 36); heavy_defo: (16, 24)
    alpha = random.uniform(1.0, 1.2)
    return kernel, sigma, alpha

if __name__ == "__main__":
    # TODO: data path
    ir_path = '../dataset/raw/ctest/Road/ir'
    vi_path = '../dataset/raw/ctest/Road/vi'
    ir_folder = pathlib.Path(ir_path)
    vi_folder = pathlib.Path(vi_path)

    ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
    vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

    # TODO: save path
    ir_affine_path          = '../dataset/raw/ctest/Road/newdata_heavy/ir_affine'
    ir_affine_deform_path   = '../dataset/raw/ctest/Road/newdata_heavy/ir_affine_deform'
    disp_affine_deform_path = '../dataset/raw/ctest/Road/newdata_heavy/disp_affine_deform'

    imfuse_affine_path        = "../dataset/raw/ctest/Road/newdata_heavy/imfuse_affine"
    imfuse_affine_deform_path = "../dataset/raw/ctest/Road/newdata_heavy/imfuse_affine_deform"

    if not os.path.exists(ir_affine_path):
        os.makedirs(ir_affine_path)
    if not os.path.exists(ir_affine_deform_path):
        os.makedirs(ir_affine_deform_path)
    if not os.path.exists(disp_affine_deform_path):
        os.makedirs(disp_affine_deform_path)

    if not os.path.exists(imfuse_affine_path):
        os.makedirs(imfuse_affine_path)
    if not os.path.exists(imfuse_affine_deform_path):
        os.makedirs(imfuse_affine_deform_path)


    for idx, (ir_file, vi_file) in enumerate(zip(ir_list, vi_list)):
        name, ext = os.path.splitext(os.path.basename(ir_file))
        filename = name + ext
        name_disp = name + '_disp.npy'

        degree, translate, scale, shear = get_random_affine_coeff()
        affine = AffineTransform(degree, translate, scale, shear)

        kernel, sigma, alpha = get_random_deformable_coeff()
        elastic = ElasticTransform(kernel_size=kernel, sigma=sigma, alpha=(alpha, alpha))

        ir = imread(str(ir_file), flags=cv2.IMREAD_GRAYSCALE) # torch.Size([1, 256, 256])
        vi = imread(str(vi_file), flags=cv2.IMREAD_GRAYSCALE)
        ir_affine, affine_disp = affine(ir.unsqueeze(0))
        ir_elastic, elastic_disp = elastic(ir_affine)

        disp = affine_disp + elastic_disp
        ir_warp = ir_elastic

        ir_affine_name        = os.path.join(ir_affine_path, filename)
        ir_affine_deform_name = os.path.join(ir_affine_deform_path, filename)
        imsave(ir_affine, ir_affine_name)
        imsave(ir_warp, ir_affine_deform_name)

        disp = disp.permute(0, 3, 1, 2)
        disp_npy = disp.data.cpu().numpy()
        disp_affine_deform_name = os.path.join(disp_affine_deform_path, name_disp)
        np.save(disp_affine_deform_name, disp_npy)

        imfuse_affine = imfuse(ir_affine, vi.unsqueeze(0)) # (256, 256, 3)
        imfuse_affine_deform = imfuse(ir_warp, vi.unsqueeze(0))

        imfuse_affine_name = os.path.join(imfuse_affine_path, filename)
        imfuse_affine_deform_name = os.path.join(imfuse_affine_deform_path, filename)
        cv2.imwrite(imfuse_affine_name, imfuse_affine)
        cv2.imwrite(imfuse_affine_deform_name, imfuse_affine_deform)

