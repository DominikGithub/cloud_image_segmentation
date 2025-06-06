'''
Cloud segmentation data loader.
'''

import os
from glob2 import glob
import numpy as np
import torch.utils.data as data
import torch
from PIL import Image, ImageOps
from skimage.filters import sobel
from skimage.feature import local_binary_pattern
import albumentations as A

base_path = "../dataset_clouds_from_lwir"


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
            A.RandomCrop(height=1024 / 2 , width=1024 / 2, p=1.0),
            A.SquareSymmetry(p=1.0),
            A.Rotate(limit=30, p=0.3),
            # image only
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(std_range=(0.05, 0.2), p=0.2),
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
        # correct TIF image orientation
        img = ImageOps.exif_transpose(img)
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = np.stack([img_np]*3, axis=-1)
        
        # # feature extraction to use all 3 RGB channels
        # features = [img_np]
        # # # Normalized
        # # norm = (img_np - np.mean(img_np)) / (np.std(img_np) + 1e-8)
        # # features.append(norm)
        # # Sobel edges
        # sobel_edges = sobel(img_np)
        # features.append(sobel_edges)
        # # LBP
        # lbp = local_binary_pattern(img_np, P=8, R=1, method='uniform')
        # features.append(lbp)
        # # # Local mean and variance
        # # local_mean = uniform_filter(img_np, size=5)
        # # # local_var = variance(img_np)
        # # # features.append([local_mean, local_var])
        # # features.append(np.array(local_mean))
        # img_np = np.stack(features, axis=0)
        
        # Mask
        mask = Image.open(mask_path)
        mask = ImageOps.exif_transpose(mask)
        mask_np = np.array(mask, dtype=np.float32) / 255.0

        # augmentation
        if   self.set_name == 'training':  augmented = self.aug_transform_train(image=img_np, mask=mask_np)
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