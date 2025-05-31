'''
Cloud segmentation data loader.
'''

import os
from glob2 import glob
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch

base_path = "../dataset_clouds_from_lwir"


class CloudSegDataloader(data.Dataset):

    def __init__(self, set_name, data_preprocessor):
        super(CloudSegDataloader, self).__init__()
        self.img_files = glob(os.path.join(base_path, set_name, 'lwir', '*.TIF'))
        self.mask_files = glob(os.path.join(base_path, set_name, 'clouds', '*.TIF'))
        self.preprocessor = data_preprocessor
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        mask_path = self.mask_files[idx]

        # image downsampling to 256x256
        img = Image.open(img_path).convert('F')
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = np.stack([img_np]*3, axis=-1)
        img_resized = Image.fromarray((img_np * 255).astype(np.uint8)).resize((256, 256), resample=Image.BILINEAR)
        img_resized_np = np.array(img_resized, dtype=np.float32) / 255.0

        enc = self.preprocessor(images=img_resized_np, return_tensors="pt")
        pixel = enc["pixel_values"].squeeze(0)

        # Mask downsampling 64x64
        mask = Image.open(mask_path)
        mask = mask.resize((64, 64), resample=Image.NEAREST)
        mask_np = np.array(mask, dtype=np.float32) / 255.0
        mask_torch_tens = torch.from_numpy(mask_np).long().unsqueeze(0)  # (1, 64, 64)
        return {
            "pixel_values": pixel,
            "labels": mask_torch_tens
        }

    def __len__(self):
        return len(self.img_files)