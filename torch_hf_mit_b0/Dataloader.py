'''

'''

import os
from glob2 import glob
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
from torchvision import transforms

base_path = "../dataset_clouds_from_lwir"


class CloudSegDataloader(data.Dataset):

    def __init__(self, set_name):
        super(CloudSegDataloader, self).__init__()
        self.img_files = glob(os.path.join(base_path, set_name, 'lwir', '*.TIF'))
        self.mask_files = glob(os.path.join(base_path, set_name, 'clouds', '*.TIF'))

    def __getitem__(self, index):
            # image
            img_path = self.img_files[index]
            data = Image.open(img_path)
            data = np.array(data) / 255.0
            # label
            mask_path = self.mask_files[index]
            label = Image.open(mask_path)
            
            # label = np.array(label)
            # print(data.shape, label.shape)

            # TODO # augmentation
            # data_transforms = transforms.Compose([transforms.RandomCrop((512,512)),
            #                      transforms.RandomRotation([+90,+180]),
            #                      transforms.RandomRotation([+180,+270]),
            #                      transforms.RandomHorizontalFlip(),
            #                      transforms.ToTensor(),
            #                    ])

            return torch.as_tensor(np.array(data).astype('float')), torch.as_tensor(np.array(label).astype('float'))

    def __len__(self):
        return len(self.img_files)