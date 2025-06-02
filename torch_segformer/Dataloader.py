'''
Cloud segmentation data loader.
'''

import os
from glob2 import glob
import numpy as np
import torch.utils.data as data
from PIL import Image
import albumentations as A

base_path = "../../dataset_clouds_from_lwir"


class CloudSegDataloader(data.Dataset):

    def __init__(self, set_name, data_preprocessor):
        super(CloudSegDataloader, self).__init__()
        self.set_name = set_name
        self.img_files = glob(os.path.join(base_path, set_name, 'lwir', '*.TIF'))
        self.mask_files = glob(os.path.join(base_path, set_name, 'clouds', '*.TIF'))
        self.preprocessor = data_preprocessor
        # augmentations
        self.aug_transform_train = A.Compose([
            # A.SmallestMaxSize(max_size=1024, p=1.0),
            A.RandomCrop(height=1024 / 4 , width=1024 / 4, p=1.0),
            A.SquareSymmetry(p=1.0),
            A.Rotate(limit=30, p=0.3),
            # image only
            # A.RandomBrightnessContrast(p=0.3),
            # A.GaussNoise(std_range=(0.1, 0.2), p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.ToTensorV2(),
        ])
        self.aug_transform_valid = A.Compose([
            # A.SmallestMaxSize(max_size=1024, p=1.0),
            A.SquareSymmetry(p=1.0),
            # A.Rotate(limit=30, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.ToTensorV2(),
        ])
        self.aug_transform_test = A.Compose([
            A.ToTensorV2(),
        ])

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        mask_path = self.mask_files[idx]
        # image
        img = Image.open(img_path).convert('F')
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = np.stack([img_np]*3, axis=-1)
        # Mask
        mask = Image.open(mask_path)
        mask_np = np.array(mask, dtype=np.float32) / 255.0

        # augmentation
        if self.set_name == 'training':    augmented = self.aug_transform_train(image=img_np, mask=mask_np)
        elif self.set_name == 'validation':augmented = self.aug_transform_valid(image=img_np, mask=mask_np)
        elif self.set_name == 'test':      augmented = self.aug_transform_test(image=img_np, mask=mask_np)
        pixel = augmented['image']
        label = augmented['mask']
        label = label.unsqueeze(0)

        # segformer image preprocessing
        enc = self.preprocessor(images=pixel, return_tensors="pt")
        pixel = enc["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel,
            "labels": label
        }

    def __len__(self):
        return len(self.img_files)