import json
import os
import pathlib
import warnings

warnings.filterwarnings("ignore")

from pathlib import PurePosixPath as GPath
from typing import Optional, Union, Text

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from scene.dataset_readers import CameraInfo
from utils.general_utils import PILtoTorch
from utils.graphics_utils import focal2fov
from utils.pose_utils_2 import smooth_camera_poses

PathType = Union[Text, pathlib.PurePosixPath]


class Camera:
    """Class to handle camera geometry."""

    def __init__(
        self,
        orientation: np.ndarray,
        position: np.ndarray,
        focal_length: Union[np.ndarray, float],
        principal_point: np.ndarray,
        image_size: np.ndarray,
        skew: Union[np.ndarray, float] = 0.0,
        pixel_aspect_ratio: Union[np.ndarray, float] = 1.0,
        radial_distortion: Optional[np.ndarray] = None,
        tangential_distortion: Optional[np.ndarray] = None,
        dtype=np.float32
    ):
        """Constructor for camera class."""
        if radial_distortion is None:
            radial_distortion = np.array([0.0, 0.0, 0.0], dtype)
        if tangential_distortion is None:
            tangential_distortion = np.array([0.0, 0.0], dtype)

        self.orientation = np.array(orientation, dtype)
        self.position = np.array(position, dtype)
        self.focal_length = np.array(focal_length, dtype)
        self.principal_point = np.array(principal_point, dtype)
        self.skew = np.array(skew, dtype)
        self.pixel_aspect_ratio = np.array(pixel_aspect_ratio, dtype)
        self.radial_distortion = np.array(radial_distortion, dtype)
        self.tangential_distortion = np.array(tangential_distortion, dtype)
        self.image_size = np.array(image_size, np.uint32)
        self.dtype = dtype

    @classmethod
    def from_json(cls, path: PathType):
        """Loads a JSON camera into memory."""
        path = GPath(path)
        # with path.open('r') as fp:
        with open(path, 'r') as fp:
            camera_json = json.load(fp)

        # Fix old camera JSON.
        if 'tangential' in camera_json:
            camera_json['tangential_distortion'] = camera_json['tangential']

        return cls(
            orientation=np.asarray(camera_json['orientation']),
            position=np.asarray(camera_json['position']),
            focal_length=camera_json['focal_length'],
            principal_point=np.asarray(camera_json['principal_point']),
            skew=camera_json['skew'],
            pixel_aspect_ratio=camera_json['pixel_aspect_ratio'],
            radial_distortion=np.asarray(camera_json['radial_distortion']),
            tangential_distortion=np.asarray(camera_json['tangential_distortion']),
            image_size=np.asarray(camera_json['image_size']),
        )

    @property
    def image_size_y(self):
        return self.image_size[1]

    @property
    def image_size_x(self):
        return self.image_size[0]

    @property
    def image_shape(self):
        return self.image_size_y, self.image_size_x


class Load_hyper_data(Dataset):
    def __init__(self, datadir, ratio=1.0, split="train"):
        datadir = os.path.expanduser(datadir)
        with open(f'{datadir}/scene.json', 'r') as f:
            scene_json = json.load(f)
        with open(f'{datadir}/metadata.json', 'r') as f:
            meta_json = json.load(f)
        with open(f'{datadir}/dataset.json', 'r') as f:
            dataset_json = json.load(f)

        self.near = scene_json['near']
        self.far = scene_json['far']
        self.coord_scale = scene_json['scale']
        self.scene_center = scene_json['center']

        self.all_img = dataset_json['ids']
        self.val_id = dataset_json['val_ids']
        self.split = split
        if len(self.val_id) == 0:
            self.i_train = np.array([i for i in np.arange(len(self.all_img)) if (i % 4 == 0)])
            self.i_test = self.i_train + 2
            self.i_test = self.i_test[:-1, ]
        else:
            self.train_id = dataset_json['train_ids']
            self.i_test = []
            self.i_train = []
            for i in range(len(self.all_img)):
                id = self.all_img[i]
                if id in self.val_id:
                    self.i_test.append(i)
                if id in self.train_id:
                    self.i_train.append(i)

        self.all_cam = [meta_json[i]['camera_id'] for i in self.all_img]
        self.all_time = [meta_json[i]['warp_id'] for i in self.all_img]
        max_time = max(self.all_time)
        self.all_time = [meta_json[i]['warp_id'] / max_time for i in self.all_img]
        self.selected_time = sorted(list(set(self.all_time)))
        self.ratio = ratio
        self.max_time = max(self.all_time)
        self.min_time = min(self.all_time)
        self.i_video = [i for i in range(len(self.all_img))]
        self.i_video.sort()
        self.all_cam_params = []
        for im in self.all_img:
            camera = Camera.from_json(f'{datadir}/camera/{im}.json')
            self.all_cam_params.append(camera)
        self.all_img_origin = self.all_img
        self.all_depth = [f'{datadir}/depth/{int(1 / ratio)}x/{i}.npy' for i in self.all_img]

        self.all_img = [f'{datadir}/rgb/{int(1 / ratio)}x/{i}.png' for i in self.all_img]

        self.h, self.w = self.all_cam_params[0].image_shape
        self.map = {}
        self.image_one = Image.open(self.all_img[0])
        self.image_one_torch = PILtoTorch(self.image_one, None).to(torch.float32)
        if os.path.exists(os.path.join(datadir, "covisible")):
            self.image_mask = [f'{datadir}/covisible/{int(2)}x/val/{i}.png' for i in self.all_img_origin]
        else:
            self.image_mask = None
        # self.generate_video_path()

    def generate_video_path(self):
        self.select_video_cams = [item for i, item in enumerate(self.all_cam_params) if i % 1 == 0]
        self.video_path, self.video_time = smooth_camera_poses(self.select_video_cams, 10)
        # breakpoint()
        self.video_path = self.video_path[:500]
        self.video_time = self.video_time[:500]
        # breakpoint()

    def __getitem__(self, index):
        if self.split == "train":
            return self.load_raw(self.i_train[index])

        elif self.split == "test":
            return self.load_raw(self.i_test[index])
        # elif self.split == "video":
        #     return self.load_video(index)

    def __len__(self):
        if self.split == "train":
            return len(self.i_train)
        elif self.split == "test":
            return len(self.i_test)
        # elif self.split == "video":
        #     return len(self.video_path)
        # return len(self.video_v2)

    def load_raw(self, idx):
        if idx in self.map.keys():
            return self.map[idx]
        camera = self.all_cam_params[idx]
        # image = Image.open(self.all_img[idx])
        image = np.array(Image.open(self.all_img[idx]))
        image = Image.fromarray(image.astype(np.uint8))
        w = image.size[0]
        h = image.size[1]
        # image = PILtoTorch(image, None)
        # image = image.to(torch.float32)[:3, :, :]
        time = self.all_time[idx]
        R = camera.orientation.T
        T = - camera.position @ R
        FovY = focal2fov(camera.focal_length, self.h)
        FovX = focal2fov(camera.focal_length, self.w)
        image_path = "/".join(self.all_img[idx].split("/")[:-1])
        image_name = self.all_img[idx].split("/")[-1]
        # if self.image_mask is not None and self.split == "test":
        #     mask = Image.open(self.image_mask[idx])
        #     mask = PILtoTorch(mask, None)
        #     mask = mask.to(torch.float32)[0:1, :, :]
        #
        #     mask = F.interpolate(mask.unsqueeze(0),
        #         size=[self.h, self.w],
        #         mode='bilinear',
        #         align_corners=False).squeeze(0)
        # else:
        #     mask = None

        caminfo = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
            image_path=image_path, image_name=image_name, width=w, height=h, fid=time)
        self.map[idx] = caminfo
        return caminfo


def format_hyper_data(data_class, split):
    if split == "train":
        data_idx = data_class.i_train
    elif split == "test":
        data_idx = data_class.i_test
    else:
        raise RuntimeError(f"Unknown split {split}")
    # dataset = data_class.copy()
    # dataset.mode = split
    cam_infos = []
    for uid, index in tqdm(enumerate(data_idx)):
        camera = data_class.all_cam_params[index]
        # image = Image.open(data_class.all_img[index])
        # image = PILtoTorch(image,None)
        time = data_class.all_time[index]
        R = camera.orientation.T
        T = - camera.position @ R
        FovY = focal2fov(camera.focal_length, data_class.h)
        FovX = focal2fov(camera.focal_length, data_class.w)
        image_path = "/".join(data_class.all_img[index].split("/")[:-1])
        image_name = data_class.all_img[index].split("/")[-1]

        # if data_class.image_mask is not None and data_class.split == "test":
        #     mask = Image.open(data_class.image_mask[index])
        #     mask = PILtoTorch(mask, None)
        #
        #     mask = mask.to(torch.float32)[0:1, :, :]
        #
        #
        # else:
        #     mask = None
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=None,
            image_path=image_path, image_name=image_name, width=int(data_class.w),
            height=int(data_class.h), fid=time
        )
        cam_infos.append(cam_info)
    return cam_infos
