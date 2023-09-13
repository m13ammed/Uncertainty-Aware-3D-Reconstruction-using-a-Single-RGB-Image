import torch.utils.data as data
from typing import Dict, Union, List, Tuple
from PIL import Image
import pyexr
import os
import random
#from datasets 
import numpy as np
import torch
#try:
#    from datasets import transforms   
#except:
#    import transforms
from torchvision import transforms
class Front3D(data.Dataset):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)
    def __init__(self, dataset_root_path: os.PathLike = '/mnt/hdd/front3d', file_list_path: os.PathLike = '/home/rashed/repos/panoptic-reconstruction/resources/front3d/train_list_3d.txt', fields: List[str] =["color", "depth"],
                 num_samples: int = 32, shuffle: bool = False, split = 'val', save_dir = '/mnt/hdd/tmp/depth/') -> None:
        super().__init__()
        height=240#224
        width=320#294
        self.output_size = (height, width)
        self.dataset_root_path = Path(dataset_root_path)
        self.save_dir = save_dir
        self.samples: List = self.load_and_filter_file_list(file_list_path)

        if shuffle:
            random.shuffle(self.samples)

        self.samples = self.samples[:num_samples]

        # Fields defines which data should be loaded
        if fields is None:
            fields = []

        self.fields = fields

        self.image_size = (320, 240)
        self.depth_image_size = (320, 240)

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


        #rgb_np, depth_np = color, depth#self.transform(color, depth)

        #to_tensor = transforms.ToTensor()
        input_tensor = color #to_tensor(rgb_np)
        input_depth = depth #to_tensor(depth_np)
        inputs = {"name": sample_path,
                  "index": index}
        inputs[("color", 0, 0)] = input_tensor
        inputs["depth_gt"] = input_depth#.unsqueeze(0)
        inputs["intrinsic"] = camera_info['intrinsic'] 
        inputs["scene_id"] = scene_id
        inputs["image_id"] = image_id
        inputs['save_dir'] = self.save_dir
        inputs['camera2world'] = camera_info['camera2world']
        inputs['blender_matrix'] = camera_info['blender_matrix']
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
            transforms.ToTensor(),
            Rotate(angle),
            transforms.Resize(224, antialias=True), # this is for computational efficiency, since rotation can be slow
            transforms.Resize(int(224/s), antialias=True),
            HorizontalFlip(do_flip),
            transforms.CenterCrop(self.output_size),
        ])

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, antialias=True),
            transforms.CenterCrop(self.output_size),
        ])

        #rgb_np = rgb_np / 255
        #transforms.ColorJitter(0.4, 0.4, 0.4),
        norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        rgb_np = norm(transform(np.array(rgb)))#(torch.Tensor(rgb)) #.transpose(2,0,1)

        depth_np = transform(np.array(depth_np))

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Resize(240, antialias=False),
            #transforms.CenterCrop(self.output_size),
        ])
        #rgb.flags['WRITEABLE'] = True
        norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        rgb_np = norm(transform(np.array(rgb)))#(torch.Tensor(rgb)) #.transpose(2,0,1)
        depth_np = transform(np.array(depth_np))

        return rgb_np, depth_np


class Rotate(object):
    """Rotates the given ``numpy.ndarray``.
    Args:
        angle (float): The rotation angle in degrees.
    """

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be rotated.
        Returns:
            img (numpy.ndarray (C x H x W)): Rotated image.
        """
        return transforms.functional.rotate(img, angle = self.angle)


class HorizontalFlip(object):
    """Rotates the given ``numpy.ndarray``.
    Args:
        angle (float): The rotation angle in degrees.
    """

    def __init__(self, do_flip):
        self.do_flip = do_flip

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be rotated.
        Returns:
            img (numpy.ndarray (C x H x W)): Rotated image.
        """

        return transforms.functional.hflip(img) if self.do_flip else img