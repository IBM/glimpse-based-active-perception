import os.path
import sys
import random
import math
import numpy as np
import time
from itertools import combinations, permutations
import builtins
from copy import deepcopy
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True
# Logging utility
from datasets.art.util import log

"""
Combinatorics:
n is total number of objects in training or test set
for training set, n will actually be (n - m)
for test set, n will actually be m
n_same_trials = nC4 * 4C1 * 3C1 * 2C2 * 2 * 2
n_diff_trials = nC5 * 5C1 * 4C2 * 2 * 2C2 * 2 * 2
"""

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


# Dimensionality of multiple-choice output
y_dim = 2
# Sequence length
seq_len = 6
# Task segmentation (for context normalization)
task_seg = [[0, 1], [2, 3], [4, 5]]



# Method for calculating number of combinations
def n_comb(n, r):
    return int(math.factorial(n) / (math.factorial(r) * math.factorial(n - r)))


# Create subsampled dataset
def subsampled_dset(shapes, n_trials):
    seq = np.array([]).astype(np.int32)
    targ = np.array([]).astype(np.int32)
    while seq.shape[0] < n_trials:
        # Sample same trial
        np.random.shuffle(shapes)
        comb = shapes[:4]
        same_targ = np.round(np.random.rand()).astype(np.int32)
        if same_targ == 0:
            same_seq = [comb[0], comb[0], comb[1], comb[1], comb[2], comb[3]]
        elif same_targ == 1:
            same_seq = [comb[0], comb[0], comb[1], comb[2], comb[3], comb[3]]
        if seq.shape[0] == 0:
            seq = np.expand_dims(np.array(same_seq), 0)
            targ = np.append(targ, same_targ)
        else:
            if not np.any(np.all(seq == np.tile(same_seq, [seq.shape[0], 1]), 1)):
                seq = np.append(seq, np.expand_dims(np.array(same_seq), 0), 0)
                targ = np.append(targ, same_targ)
            else:
                sample_again = True
                while sample_again:
                    # Sample another same trial
                    np.random.shuffle(shapes)
                    comb = shapes[:4]
                    same_targ = np.round(np.random.rand()).astype(np.int32)
                    if same_targ == 0:
                        same_seq = [comb[0], comb[0], comb[1], comb[1], comb[2], comb[3]]
                    elif same_targ == 1:
                        same_seq = [comb[0], comb[0], comb[1], comb[2], comb[3], comb[3]]
                    if not np.any(np.all(seq == np.tile(same_seq, [seq.shape[0], 1]), 1)):
                        sample_again = False
                        seq = np.append(seq, np.expand_dims(np.array(same_seq), 0), 0)
                        targ = np.append(targ, same_targ)
        # Sample different trial
        np.random.shuffle(shapes)
        comb = shapes[:5]
        diff_targ = np.round(np.random.rand()).astype(np.int32)
        if diff_targ == 0:
            diff_seq = [comb[0], comb[1], comb[2], comb[3], comb[4], comb[4]]
        elif diff_targ == 1:
            diff_seq = [comb[0], comb[1], comb[2], comb[2], comb[3], comb[4]]
        if not np.any(np.all(seq == np.tile(diff_seq, [seq.shape[0], 1]), 1)):
            seq = np.append(seq, np.expand_dims(np.array(diff_seq), 0), 0)
            targ = np.append(targ, diff_targ)
        else:
            sample_again = True
            while sample_again:
                # Sample another same trial
                np.random.shuffle(shapes)
                comb = shapes[:5]
                diff_targ = np.round(np.random.rand()).astype(np.int32)
                if diff_targ == 0:
                    diff_seq = [comb[0], comb[1], comb[2], comb[3], comb[4], comb[4]]
                elif diff_targ == 1:
                    diff_seq = [comb[0], comb[1], comb[2], comb[2], comb[3], comb[4]]
                if not np.any(np.all(seq == np.tile(diff_seq, [seq.shape[0], 1]), 1)):
                    sample_again = False
                    seq = np.append(seq, np.expand_dims(np.array(diff_seq), 0), 0)
                    targ = np.append(targ, diff_targ)
    # Shuffle
    trial_order = np.arange(len(seq))
    np.random.shuffle(trial_order)
    seq = seq[trial_order, :]
    targ = targ[trial_order]
    # Select subset
    seq = seq[:n_trials, :]
    targ = targ[:n_trials]
    return seq, targ


# Create full dataset
def full_dset(shapes, n_trials):
    # All same trials
    all_same_seq = []
    all_same_targ = []
    all_same_trial_comb = builtins.list(combinations(shapes, 4))
    for comb in all_same_trial_comb:
        comb = builtins.list(comb)
        for s1 in comb:
            comb_minus_s1 = deepcopy(comb)
            comb_minus_s1.remove(s1)
            source_same_pair = [s1, s1]
            for s2 in comb_minus_s1:
                diff_pair = deepcopy(comb_minus_s1)
                diff_pair.remove(s2)
                diff_pair_perm = builtins.list(permutations(diff_pair, 2))
                for diff_pair in diff_pair_perm:
                    same_pair = [s2, s2]
                    diff_pair = builtins.list(diff_pair)
                    choices = [same_pair, diff_pair]
                    choices_perm = builtins.list(permutations(choices, 2))
                    for targ, choices in enumerate(choices_perm):
                        choices = builtins.list(choices)
                        same_seq = source_same_pair + choices[0] + choices[1]
                        all_same_seq.append(same_seq)
                        all_same_targ.append(targ)
    # All different trials
    all_diff_seq = []
    all_diff_targ = []
    all_diff_trial_comb = builtins.list(combinations(shapes, 5))
    for comb in all_diff_trial_comb:
        comb = builtins.list(comb)
        all_diff1_comb = builtins.list(combinations(comb, 2))
        for diff1_comb in all_diff1_comb:
            diff1_comb = builtins.list(diff1_comb)
            comb_minus_diff1 = deepcopy(comb)
            comb_minus_diff1.remove(diff1_comb[0])
            comb_minus_diff1.remove(diff1_comb[1])
            all_diff1_perm = builtins.list(permutations(diff1_comb, 2))
            for diff1_perm in all_diff1_perm:
                source_diff_pair = builtins.list(diff1_perm)
                all_diff2_comb = builtins.list(combinations(comb_minus_diff1, 2))
                for diff2_comb in all_diff2_comb:
                    diff2_comb = builtins.list(diff2_comb)
                    s = deepcopy(comb_minus_diff1)
                    s.remove(diff2_comb[0])
                    s.remove(diff2_comb[1])
                    same_pair = [s[0], s[0]]
                    all_diff2_perm = builtins.list(permutations(diff2_comb, 2))
                    for diff2_perm in all_diff2_perm:
                        diff_pair = builtins.list(diff2_perm)
                        choices = [diff_pair, same_pair]
                        choices_perm = builtins.list(permutations(choices, 2))
                        for targ, choices in enumerate(choices_perm):
                            choices = builtins.list(choices)
                            diff_seq = source_diff_pair + choices[0] + choices[1]
                            all_diff_seq.append(diff_seq)
                            all_diff_targ.append(targ)
    # Duplicate trials if necessary (so that trial types are balanced)
    # Same trials
    if len(all_same_seq) < n_trials / 2:
        all_same_seq_augmented = deepcopy(all_same_seq)
        all_same_targ_augmented = deepcopy(all_same_targ)
        for a in range(int(n_trials / 2) - len(all_same_seq)):
            trial_ind = np.floor(np.random.rand() * len(all_same_seq)).astype(np.int32)
            all_same_seq_augmented.append(all_same_seq[trial_ind])
            all_same_targ_augmented.append(all_same_targ[trial_ind])
        all_same_seq = all_same_seq_augmented
        all_same_targ = all_same_targ_augmented
    # Different trials
    if len(all_diff_seq) < n_trials / 2:
        all_diff_seq_augmented = deepcopy(all_diff_seq)
        all_diff_targ_augmented = deepcopy(all_diff_targ)
        for a in range(int(n_trials / 2) - len(all_diff_seq)):
            trial_ind = np.floor(np.random.rand() * len(all_diff_seq)).astype(np.int32)
            all_diff_seq_augmented.append(all_diff_seq[trial_ind])
            all_diff_targ_augmented.append(all_diff_targ[trial_ind])
        all_diff_seq = all_diff_seq_augmented
        all_diff_targ = all_diff_targ_augmented
    # Combine same and different trials
    seq = all_same_seq + all_diff_seq
    targ = all_same_targ + all_diff_targ
    # Shuffle
    trial_order = np.arange(len(seq))
    np.random.shuffle(trial_order)
    seq = np.array(seq)[trial_order, :]
    targ = np.array(targ)[trial_order]
    # Select subset
    seq = seq[:n_trials, :]
    targ = targ[:n_trials]
    return seq, targ


# Task generator
def create_task(
        n_shapes, m_holdout, train_set_size, test_set_size, train_gen_method, test_gen_method, train_proportion,
        train_shapes, test_shapes):
    log.info('n_shapes = ' + str(n_shapes) + '...')
    log.info('m_holdout = ' + str(m_holdout) + '...')
    # If m = 0, training and test sets are drawn from same set of shapes
    if m_holdout == 0:
        # Total number of possible trials
        shapes_avail = n_shapes
        n_same_trials = n_comb(shapes_avail, 4) * n_comb(4, 1) * n_comb(3, 1) * n_comb(2, 2) * 2 * 2
        n_diff_trials = n_comb(shapes_avail, 5) * n_comb(5, 2) * 2 * n_comb(3, 2) * 2 * n_comb(1, 1) * 2
        total_unique_trials = n_same_trials + n_diff_trials
        log.info('Total possible trials = ' + str(total_unique_trials) + '...')
        if n_diff_trials > n_same_trials:
            total_trials = n_diff_trials * 2
        else:
            total_trials = n_same_trials * 2
        if train_set_size + test_set_size > total_trials:
            # Proportion of training set size vs. test set size
            train_proportion = train_proportion
            test_proportion = 1 - train_proportion
            # Create training/test set sizes
            log.info('Desired training set size (' + str(train_set_size) + ') and test set size (' + str(
                test_set_size) + ') combined are larger than total number of possible trials for this task (' + str(
                total_trials) + ')...')
            train_set_size = np.round(train_proportion * total_trials).astype(np.int32)
            log.info('Changing training set size to ' + str(train_set_size) + '...')
            test_set_size = np.round(test_proportion * total_trials).astype(np.int32)
            log.info('Changing test set size to ' + str(test_set_size) + '...')
        else:
            log.info('Training set size = ' + str(train_set_size) + '...')
            log.info('Test set size = ' + str(test_set_size) + '...')
    # Otherwise, training and test sets are completely disjoint (in terms of the shapes that are used)
    else:
        # Ensure that there are enough potential trials for desired training set size (or change training set size)
        shapes_avail = n_shapes - m_holdout
        n_same_trials = n_comb(shapes_avail, 4) * n_comb(4, 1) * n_comb(3, 1) * n_comb(2, 2) * 2 * 2
        n_diff_trials = n_comb(shapes_avail, 5) * n_comb(5, 2) * 2 * n_comb(3, 2) * 2 * n_comb(1, 1) * 2
        total_unique_trials = n_same_trials + n_diff_trials
        log.info('Total possible training trials = ' + str(total_unique_trials) + '...')
        if n_diff_trials > n_same_trials:
            total_trials = n_diff_trials * 2
        else:
            total_trials = n_same_trials * 2
        if train_set_size > total_trials:
            log.info('Desired training set size (' + str(
                train_set_size) + ') is larger than total number of possible training trials for this task (' + str(
                total_trials) + ')...')
            log.info('Changing training set size to ' + str(total_trials) + '...')
            train_set_size = total_trials
        else:
            log.info('Training set size = ' + str(train_set_size) + '...')
        # Ensure that there are enough potential trials for desired test set size (or change test set size)
        shapes_avail = m_holdout
        n_same_trials = n_comb(shapes_avail, 4) * n_comb(4, 1) * n_comb(3, 1) * n_comb(2, 2) * 2 * 2
        n_diff_trials = n_comb(shapes_avail, 5) * n_comb(5, 2) * 2 * n_comb(3, 2) * 2 * n_comb(1, 1) * 2
        total_unique_trials = n_same_trials + n_diff_trials
        log.info('Total possible test trials = ' + str(total_unique_trials) + '...')
        if n_diff_trials > n_same_trials:
            total_trials = n_diff_trials * 2
        else:
            total_trials = n_same_trials * 2
        if test_set_size > total_trials:
            log.info('Desired test set size (' + str(
                test_set_size) + ') is larger than total number of possible test trials for this task (' + str(
                total_trials) + ')...')
            log.info('Changing test set size to ' + str(total_trials) + '...')
            test_set_size = total_trials
        else:
            log.info('Test set size = ' + str(test_set_size) + '...')

    # Create all possible trials
    if m_holdout == 0:
        if train_gen_method == 'subsample':
            all_seq, all_targ = subsampled_dset(train_shapes, train_set_size + test_set_size)
        elif train_gen_method == 'full_space':
            all_seq, all_targ = full_dset(train_shapes, train_set_size + test_set_size)
        # Split train and test sets
        train_seq = all_seq[:train_set_size, :]
        train_targ = all_targ[:train_set_size]
        test_seq = all_seq[train_set_size:, :]
        test_targ = all_targ[train_set_size:]
    # Otherwise, training and test sets are completely disjoint (in terms of the shapes that are used), and can be generated separately
    else:
        if train_gen_method == 'subsample':
            train_seq, train_targ = subsampled_dset(train_shapes, train_set_size)
        elif train_gen_method == 'full_space':
            train_seq, train_targ = full_dset(train_shapes, train_set_size)
        if test_gen_method == 'subsample':
            test_seq, test_targ = subsampled_dset(test_shapes, test_set_size)
        elif test_gen_method == 'full_space':
            test_seq, test_targ = full_dset(test_shapes, test_set_size)

    # Create training and test sets
    train_set = {'seq_ind': train_seq, 'y': train_targ}
    test_set = {'seq_ind': test_seq, 'y': test_targ}

    return train_set, test_set


class RMTSdataset(Dataset):
    def __init__(self, root_dir, seq_set, img_size, n_shapes, transformations=None):
        self.root_dir = root_dir

        # task_gen = __import__(args.task)
        # log.info('Generating task: ' + args.task + '...')

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
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
        self.seq_len = 2

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

        x_in = []
        for m in [0, 2]:
            full_img = 255 * np.ones((160, 160))
            transx, transy = np.random.randint(-5, 5, (2,))

            full_img = fill_image(self.all_imgs, seq_ind[0], full_img, [40 + transx, 40 + transy])
            transx, transy = np.random.randint(-5, 5, (2,))

            full_img = fill_image(self.all_imgs, seq_ind[1], full_img, [40 + transx, 120 + transy])
            transx, transy = np.random.randint(-5, 5, (2,))

            full_img = fill_image(self.all_imgs, seq_ind[2 + m], full_img, [120 + transx, 40 + transy])
            transx, transy = np.random.randint(-5, 5, (2,))

            full_img = fill_image(self.all_imgs, seq_ind[3 + m], full_img, [120 + transx, 120 + transy])

            # x_in1 = np.concatenate([x_seq[0,:,:], x_seq[1,:,:], x_seq[2,:,:]], axis=1)
            # x_in2 = np.concatenate([x_seq[3,:,:], x_seq[4,:,:], x_seq[5+m,:,:]], axis=1)
            x_in.append(full_img)
        # x_in.append(np.concatenate([x_in1, x_in2], axis=0))

        # img = [TF.rotate(TF.adjust_brightness(self.transforms(Image.fromarray(image[i].astype(np.uint8))),brightness_factor),angle=angle) for i in range(16)]
        img = [self.transforms(Image.fromarray(x_in[i].astype(np.uint8))) for i in range(len(x_in))]

        # img = [self.transforms(Image.fromarray(image[i].astype(np.uint8))) for i in range(16)]

        # img = [self.transforms(Image.fromarray(image[i].astype(np.uint8))) for i in range(16)]

        # resize_image.append(misc.imresize(image[idx,:,:], (self.img_size, self.img_size)))
        resize_image = torch.stack(img, dim=0)

        return resize_image, y

