import sys
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True
# Logging utility
from datasets.art.util import log

"""
Combinatorics:
For n objects, there are n^2 possible combinations, n of which will be 'same' trials.
To balance 'same' and 'different' trials, create copies of 'same' trials.
Number of 'different' trials = n * (n - 1).
Number of 'same' trials = n.
To balance trial types, create ((n * (n - 1)) / n) copies of 'same' trials.
Number of total possible trials = (n * (n - 1)) * 2.
For training set, n will actually be (n - m).
For test set, n will actually be m.
"""

# Dimensionality of multiple-choice output
y_dim = 2
# Sequence length
seq_len = 2


# Task generator
def create_task(
        n_shapes, m_holdout, train_set_size, test_set_size, train_gen_method, test_gen_method, train_proportion,
        train_shapes, test_shapes
):
    log.info('n_shapes = ' + str(n_shapes) + '...')
    log.info('m_holdout = ' + str(m_holdout) + '...')
    # If m = 0, training and test sets are drawn from same set of shapes
    if m_holdout == 0:
        # Total number of possible trials
        shapes_avail = n_shapes
        total_trials = (shapes_avail * (shapes_avail - 1)) * 2
        log.info('Total possible trials = ' + str(total_trials) + '...')
        # Proportion of training set size vs. test set size
        train_proportion = train_proportion
        test_proportion = 1 - train_proportion
        # Create training/test set sizes
        train_set_size = np.round(train_proportion * total_trials).astype(np.int32)
        test_set_size = np.round(test_proportion * total_trials).astype(np.int3232)
        log.info('Training set size = ' + str(train_set_size) + '...')
        log.info('Test set size = ' + str(test_set_size) + '...')
    # Otherwise, training and test sets are completely disjoint (in terms of the shapes that are used)
    else:
        # Ensure that there are enough potential trials for desired training set size (or change training set size)
        shapes_avail = n_shapes - m_holdout
        total_trials = (shapes_avail * (shapes_avail - 1)) * 2
        log.info('Total possible training trials = ' + str(total_trials) + '...')
        if train_set_size > total_trials:
            log.info('Desired training set size (' + str(
                train_set_size) + ') is larger than total number of possible training trials for this task (' + str(
                total_trials) + ')...')
            log.info('Changing training set size to ' + str(total_trials) + '...')
            train_set_size = total_trials
        else:
            log.info('Training set size = ' + str(train_set_size) + '...')
        # Ensure that there are enough potential trials for desired test set size (or change test set size)
        shapes_avail = n_shapes - (n_shapes - m_holdout)
        total_trials = (shapes_avail * (shapes_avail - 1)) * 2
        log.info('Total possible test trials = ' + str(total_trials) + '...')
        if test_set_size > total_trials:
            log.info('Desired test set size (' + str(
                test_set_size) + ') is larger than total number of possible test trials for this task (' + str(
                total_trials) + ')...')
            log.info('Changing test set size to ' + str(total_trials) + '...')
            test_set_size = total_trials
        else:
            log.info('Test set size = ' + str(test_set_size) + '...')

    # If m = 0, training and test sets are drawn from same set of shapes
    if m_holdout == 0:
        # Create all possible trials
        same_trials = []
        diff_trials = []
        for shape1 in train_shapes:
            for shape2 in train_shapes:
                if shape1 == shape2:
                    same_trials.append([shape1, shape2])
                else:
                    diff_trials.append([shape1, shape2])
        # Shuffle
        random.shuffle(same_trials)
        random.shuffle(diff_trials)
        # Split trials for train and test sets
        same_trials_train = same_trials[:np.round(train_proportion * len(same_trials)).astype(np.int32)]
        same_trials_test = same_trials[np.round(train_proportion * len(same_trials)).astype(np.int32):]
        diff_trials_train = diff_trials[:np.round(train_proportion * len(diff_trials)).astype(np.int32)]
        diff_trials_test = diff_trials[np.round(train_proportion * len(diff_trials)).astype(np.int32):]
    # Otherwise, training and test sets are completely disjoint (in terms of the shapes that are used), and can be generated separately
    else:
        # Create all possible training trials
        same_trials_train = []
        diff_trials_train = []
        for shape1 in train_shapes:
            for shape2 in train_shapes:
                if shape1 == shape2:
                    same_trials_train.append([shape1, shape2])
                else:
                    diff_trials_train.append([shape1, shape2])
        # Shuffle
        random.shuffle(same_trials_train)
        random.shuffle(diff_trials_train)
        # Create all possible test trials
        same_trials_test = []
        diff_trials_test = []
        for shape1 in test_shapes:
            for shape2 in test_shapes:
                if shape1 == shape2:
                    same_trials_test.append([shape1, shape2])
                else:
                    diff_trials_test.append([shape1, shape2])
        # Shuffle
        random.shuffle(same_trials_test)
        random.shuffle(diff_trials_test)
    # Duplicate 'same' trials to match number of 'different' trials
    same_trials_train_balanced = []
    for t in range(len(diff_trials_train)):
        same_trials_train_balanced.append(
            same_trials_train[np.floor(np.random.rand() * len(same_trials_train)).astype(np.int32)])
    same_trials_test_balanced = []
    for t in range(len(diff_trials_test)):
        same_trials_test_balanced.append(
            same_trials_test[np.floor(np.random.rand() * len(same_trials_test)).astype(np.int32)])
    # Combine all same and different trials for training set
    all_train_seq = []
    all_train_targ = []
    for t in range(len(same_trials_train_balanced)):
        all_train_seq.append(same_trials_train_balanced[t])
        all_train_targ.append(0)
    for t in range(len(diff_trials_train)):
        all_train_seq.append(diff_trials_train[t])
        all_train_targ.append(1)
    # Combine all same and different trials for test set
    all_test_seq = []
    all_test_targ = []
    for t in range(len(same_trials_test_balanced)):
        all_test_seq.append(same_trials_test_balanced[t])
        all_test_targ.append(0)
    for t in range(len(diff_trials_test)):
        all_test_seq.append(diff_trials_test[t])
        all_test_targ.append(1)
    # Shuffle trials in training set
    train_ind = np.arange(len(all_train_seq))
    np.random.shuffle(train_ind)
    all_train_seq = np.array(all_train_seq)[train_ind]
    all_train_targ = np.array(all_train_targ)[train_ind]
    # Shuffle trials in test set
    test_ind = np.arange(len(all_test_seq))
    np.random.shuffle(test_ind)
    all_test_seq = np.array(all_test_seq)[test_ind]
    all_test_targ = np.array(all_test_targ)[test_ind]
    # Select subset if desired dataset size is smaller than number of all possible trials
    if (train_set_size + test_set_size) < total_trials:
        all_train_seq = all_train_seq[:train_set_size, :]
        all_train_targ = all_train_targ[:train_set_size]
        all_test_seq = all_test_seq[:test_set_size, :]
        all_test_targ = all_test_targ[:test_set_size]

    # Create training and test sets
    train_set = {'seq_ind': all_train_seq, 'y': all_train_targ}
    test_set = {'seq_ind': all_test_seq, 'y': all_test_targ}

    return train_set, test_set


def fill_image(all_imgs, idx, full_img, p):
    h = all_imgs[idx].shape[0]
    w = all_imgs[idx].shape[1]
    # print(h,w)
    x = p[0]
    y = p[1]
    if h % 2 == 0 and w % 2 != 0:
        full_img[x - h // 2:x + h // 2, y - w // 2:y + w // 2 + 1] = all_imgs[idx]
    elif h % 2 != 0 and w % 2 == 0:
        full_img[x - h // 2:x + h // 2 + 1, y - w // 2:y + w // 2] = all_imgs[idx]
    elif h % 2 != 0 and w % 2 != 0:
        full_img[x - h // 2:x + h // 2 + 1, y - w // 2:y + w // 2 + 1] = all_imgs[idx]
    else:
        full_img[x - h // 2:x + h // 2, y - w // 2:y + w // 2] = all_imgs[idx]

    return full_img


class SDdataset(Dataset):
    def __init__(self, root_dir, seq_set, img_size, n_shapes, transformations=None):
        self.root_dir = root_dir
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Lambda(lambda X: 2 * X - 1.0),
                transforms.Resize((img_size, img_size)),

            ] + (transformations if transformations is not None else [])
        )

        self.img_size = img_size
        self.all_imgs = []
        for i in range(n_shapes):
            img_fname = os.path.join(root_dir, 'resizedcropped' + str(i) + '.npy')
            img = np.load(img_fname)
            # img = torch.Tensor(np.array(Image.open(img_fname))) / 255.
            self.all_imgs.append(img)
        # self.all_imgs = np.stack(self.all_imgs,axis= 0)
        # print(self.all_imgs.shape)

        self.seq_ind = seq_set['seq_ind']
        self.y = seq_set['y']
        self.len = self.seq_ind.shape[0]

    # print(self.file_names[:100])
    # if dataset_type == 'train':
    # 	self.file_names = self.file_names[:10000]

    # self.embeddings = np.load('./embedding.npy')

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # data_path = os.path.join(self.root_dir, self.file_names[idx])
        seq_ind = self.seq_ind[idx]
        y = self.y[idx]

        # x_seq = self.all_imgs[seq_ind]
        # print(x_seq.shape)
        # print(seq_ind.shape,x_seq.shape)

        full_img = 255 * np.ones((160, 160))
        transx, transy = np.random.randint(-5, 5, (2,))

        full_img = fill_image(self.all_imgs, seq_ind[0], full_img, [80 + transx, 40 + transy])
        transx, transy = np.random.randint(-5, 5, (2,))

        full_img = fill_image(self.all_imgs, seq_ind[1], full_img, [80 + transx, 120 + transy])

        # x_in1 = np.concatenate([x_seq[0,:,:], x_seq[1,:,:], x_seq[2,:,:]], axis=1)
        # x_in2 = np.concatenate([x_seq[3,:,:], x_seq[4,:,:], x_seq[5+m,:,:]], axis=1)

        # x_in.append(np.concatenate([x_in1, x_in2], axis=0))

        # img = [TF.rotate(TF.adjust_brightness(self.transforms(Image.fromarray(image[i].astype(np.uint8))),brightness_factor),angle=angle) for i in range(16)]
        img = self.transforms(Image.fromarray(full_img.astype(np.uint8)))

        # img = [self.transforms(Image.fromarray(image[i].astype(np.uint8))) for i in range(16)]

        # img = [self.transforms(Image.fromarray(image[i].astype(np.uint8))) for i in range(16)]

        # resize_image.append(misc.imresize(image[idx,:,:], (self.img_size, self.img_size)))

        return img, y
