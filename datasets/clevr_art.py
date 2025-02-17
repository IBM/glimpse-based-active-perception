from typing import Literal
import numpy as np
import os
import torch
import torchvision
from torch.utils.data import Dataset
import glob
import glob
from torchvision.transforms import transforms


class ClevrARTdataset(Dataset):
    def __init__(self, root_dir, name: Literal['rmts', 'idrules'], type, img_size, custom_trafos=[], custom_dataset_name=''):
        self.root_dir = root_dir
        self.transforms = transforms.Compose(
            [
                transforms.Lambda(lambda X: X / 255),
                transforms.Resize((img_size, img_size)),
                *custom_trafos
            ]
        )

        self.img_size = img_size
        self.allprobs = glob.glob(os.path.join(root_dir, f'{name}_images', type + '_ood' + custom_dataset_name) + '/prob_*')

        self.len = len(self.allprobs)

        if name == 'rmts':
            label_file_prefix = 'RMTS'
            self.seq_len = 2
        elif name == 'idrules':
            label_file_prefix = 'identity_rules'
            self.seq_len = 4
        else:
            raise NotImplementedError

        self.alltargets = np.load(os.path.join(root_dir, f'{name}_images', f'{label_file_prefix}_ood_{type}.npz'))['y']

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        prob_file = self.allprobs[idx]
        y = self.alltargets[int(prob_file.split('/')[-1].split('_')[1])]

        img = []
        for m in range(self.seq_len):
            img.append(
                self.transforms(
                    torchvision.io.read_image(os.path.join(prob_file, f'CLEVR_{m}.png'), torchvision.io.ImageReadMode.RGB))
            )

        resize_image = torch.stack(img, dim=0)

        return resize_image, y
