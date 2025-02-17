import os
import glob
import torch
from torch.utils.data import Dataset
import torchvision


class SVRTFull(Dataset):
    def __init__(
            self,
            root,
            train,
            task_idx,
            resize=None,
            transform=None,
            num_samples=None,
            rgb_format=False,
    ):
        """
        labelling convention: 1 corresponds to "same", 0 corresponds to "different"
        """
        super().__init__()

        if resize is not None:
            self.resize = torchvision.transforms.Resize(resize)
        else:
            self.resize = None

        self.transform = transform
        self.data_root = os.path.join(root, f'svrt/svrt_dataset_full/results_problem_{task_idx}')
        self.file_names = glob.glob(f'{self.data_root}/*.png')
        self.file_names = sorted(self.file_names, key=self.get_img_number)

        if train:
            if num_samples is None:
                self.num_samples = 50000
            else:
                self.num_samples = num_samples
            self.file_names = self.file_names[:self.num_samples]
        else:
            if num_samples is None:
                self.num_samples = 10000
            else:
                self.num_samples = num_samples

            self.file_names = self.file_names[-self.num_samples:]

        self.rgb_format = rgb_format

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img = torchvision.io.read_image(
            self.file_names[idx], 
            torchvision.io.ImageReadMode.RGB if self.rgb_format else torchvision.io.ImageReadMode.GRAY)
        img = 1 - img / 255

        if self.resize is not None:
            img = self.resize(img)
            img[img > 0.] = 1.
        if self.transform is not None:
            img = self.transform(img)

        label = self.get_label(self.file_names[idx])
        return img, torch.tensor(label)

    @staticmethod
    def get_img_number(img_path):
        return int(img_path.split('/')[-1].split('.png')[0].split('_')[-1])

    @staticmethod
    def get_label(img_path):
        return int(img_path.split('/')[-1].split('.png')[0].split('_')[1])

class OOD_SVRTFull(SVRTFull):
    def __init__(
            self,
            subset_name,
            root,
            resize=None,
            transform=None,
            num_samples=None,
            take_first=True,
            rgb_format=False,
    ):
        super().__init__(
            root, False, 1, resize, transform, num_samples, 
            rgb_format=rgb_format)

        self.data_root = os.path.join(root, f'svrt/svrt_task_1_ood_original_res/{subset_name}_test')
        self.file_names = glob.glob(f'{self.data_root}/*.png')
        self.file_names = sorted(self.file_names, key=self.get_img_number)

        if num_samples is None:
            self.num_samples = 5600
        else:
            self.num_samples = num_samples
        assert self.num_samples <= 5600

        if take_first:
            self.file_names = self.file_names[:self.num_samples]
        else:
            self.file_names = self.file_names[-self.num_samples:]
