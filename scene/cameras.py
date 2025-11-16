#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from torch import nn
import numpy as np
from PIL import Image
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image_path, depth_path,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False,
                 is_nerf_synthetic = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path
        self.depth_path = depth_path
        self.depth_params = depth_params
        self.train_test_exp = train_test_exp
        self.is_test_dataset = is_test_dataset
        self.is_test_view = is_test_view
        self.is_nerf_synthetic = is_nerf_synthetic
        self.resolution = resolution

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self._original_image = None
        self._alpha_mask = None
        self._invdepthmap = None
        self._depth_mask = None

        self.image_width = resolution[0]
        self.image_height = resolution[1]

        self.depth_reliable = self._assess_depth_reliability()

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def _assess_depth_reliability(self):
        if not self.depth_path or not os.path.exists(self.depth_path):
            return False
        if self.depth_params is None:
            return True
        scale = self.depth_params.get("scale", 0.0)
        med_scale = self.depth_params.get("med_scale", 0.0)
        if scale < 0.2 * med_scale or scale > 5 * med_scale:
            return False
        return True

    def _ensure_image_loaded(self):
        if self._original_image is not None:
            return
        with Image.open(self.image_path) as img:
            resized_image_rgb = PILtoTorch(img, self.resolution).float()
        gt_image = resized_image_rgb[:3, ...]
        if resized_image_rgb.shape[0] == 4:
            alpha_mask = resized_image_rgb[3:4, ...]
        else:
            alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...])

        if self.train_test_exp and self.is_test_view:
            if self.is_test_dataset:
                alpha_mask[..., :alpha_mask.shape[-1] // 2] = 0
            else:
                alpha_mask[..., alpha_mask.shape[-1] // 2:] = 0

        self._alpha_mask = alpha_mask.to(self.data_device)
        self._original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)

    def _ensure_depth_loaded(self):
        if self._invdepthmap is not None or not self.depth_reliable:
            return
        self._ensure_image_loaded()
        if not self.depth_path:
            self.depth_reliable = False
            return

        try:
            invdepth = cv2.imread(self.depth_path, -1)
            if invdepth is None:
                raise FileNotFoundError(f"Depth file '{self.depth_path}' could not be read.")
            invdepth = invdepth.astype(np.float32)
            denom = 512 if self.is_nerf_synthetic else float(2**16)
            invdepth = invdepth / denom
            invdepth = cv2.resize(invdepth, self.resolution)
            invdepth[invdepth < 0] = 0

            if self.depth_params is not None and self.depth_params.get("scale", 0) > 0:
                invdepth = invdepth * self.depth_params["scale"] + self.depth_params["offset"]

            if invdepth.ndim != 2:
                invdepth = invdepth[..., 0]

            self._invdepthmap = torch.from_numpy(invdepth[None]).to(self.data_device)
            self._depth_mask = torch.ones_like(self._alpha_mask)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            self.depth_reliable = False
        except IOError:
            print(f"Error: Unable to open the image file '{self.depth_path}'. It may be corrupted or an unsupported format.")
            self.depth_reliable = False
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {self.depth_path}: {e}")
            self.depth_reliable = False

    def release_data(self):
        self._original_image = None
        self._alpha_mask = None
        self._invdepthmap = None
        self._depth_mask = None

    @property
    def original_image(self):
        self._ensure_image_loaded()
        return self._original_image

    @property
    def alpha_mask(self):
        self._ensure_image_loaded()
        return self._alpha_mask

    @property
    def invdepthmap(self):
        self._ensure_depth_loaded()
        return self._invdepthmap

    @property
    def depth_mask(self):
        self._ensure_depth_loaded()
        return self._depth_mask
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
