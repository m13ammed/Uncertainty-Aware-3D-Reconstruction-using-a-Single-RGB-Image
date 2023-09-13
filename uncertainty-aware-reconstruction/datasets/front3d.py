import torch.utils.data as data
from pathlib import Path
from typing import Dict, Union, List, Tuple
from PIL import Image
import pyexr
import os
import random
#from datasets 
import numpy as np

try:
    from datasets import transforms   
except:
    import transforms

class Front3D(data.Dataset):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)
    def __init__(self, dataset_root_path: os.PathLike = '/mnt/hdd/front3d', file_list_path: os.PathLike = '/home/rashed/repos/panoptic-reconstruction/resources/front3d/train_list_3d.txt', fields: List[str] =["color", "depth"],
                 num_samples: int = 32, shuffle: bool = False, split = 'val', save_dir = '/mnt/hdd/tmp/depth/') -> None:
        super().__init__()
        height=224
        width=288
        self.output_size = (height, width)
        self.dataset_root_path = Path(dataset_root_path)

        self.samples: List = self.load_and_filter_file_list(file_list_path)

        if shuffle:
            random.shuffle(self.samples)

        self.samples = self.samples[:num_samples]

        # Fields defines which data should be loaded
        if fields is None:
            fields = []

        self.fields = fields

        self.image_size = (320, 240)
        self.depth_image_size = (160, 120)

        self.depth_min = 0 #config.MODEL.PROJECTION.DEPTH_MIN
        self.depth_max = 20 #config.MODEL.PROJECTION.DEPTH_MAX


        if split == 'train':
            self.transform = self.train_transform
        elif split == 'holdout':
            self.transform = self.val_transform
        elif split == 'val':
            self.transform = self.val_transform
        else:
            raise (RuntimeError("Invalid dataset split: " + split + "\n"
                                "Supported dataset splits are: train, val"))

        self.save_dir = save_dir
    def __getitem__(self, index) -> Dict:
        sample_path = self.samples[index]
        scene_id = sample_path.split("/")[0]
        image_id = sample_path.split("/")[1]

        try:

            # 2D data
            color = Image.open(self.dataset_root_path / scene_id / f"rgb_{image_id}.png", formats=["PNG"])

            depth = pyexr.read(str(self.dataset_root_path / scene_id / f"depth_{image_id}.exr")).squeeze().copy()#[::-1, ::-1].copy()

            color, depth = self.transform(np.asarray(color), depth)
            camera_info = np.load(self.dataset_root_path / scene_id / f"campose_{image_id}.npz")
        except Exception as e:
            print(sample_path)
            print(e)


        rgb_np, depth_np = color, depth#color, depth#self.transform(color, depth)

        to_tensor = transforms.ToTensor()
        input_tensor = to_tensor(rgb_np)
        input_depth = to_tensor(depth_np)
        inputs = {"name": sample_path,
                  "index": index}
        inputs[("color", 0, 0)] = input_tensor
        inputs["depth_gt"] = input_depth.unsqueeze(0)
        inputs["intrinsic"] = camera_info['intrinsic'] 
        inputs["scene_id"] = scene_id
        inputs["image_id"] = image_id
        inputs['save_dir'] = self.save_dir
        return inputs
    

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def load_and_filter_file_list(file_list_path: os.PathLike) -> List[str]:
        with open(file_list_path) as f:
            content = f.readlines()

        images = [line.strip() for line in content]

        return images

    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5) # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(240.0 / self.image_size[1]), # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.HorizontalFlip(do_flip),
            transforms.CenterCrop(self.output_size),
        ])

        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np) # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Resize(240.0/ self.image_size[1] ),
            transforms.CenterCrop(self.output_size),
        ])

        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np


