import sys
import random
import math
import numpy as np
import time
from itertools import combinations, permutations
import builtins
from copy import deepcopy
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from datasets.art.util import log


# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True

"""
Combinatorics:
n is total number of objects in training or test set
for training set, n will actually be (n - m)
for test set, n will actually be m
n_AAA_trials = nC4 * 4C1 * 3C1 * 4P4
n_ABA_trials = nC4 * 4C1 * 3C1 * 2C1 * 1C1 * 4P$
n_ABB_trials = nC4 * 4C1 * 3C1 * 2C1 * 1C1 * 4P$
n_trials_total = n_AAA_trials + n_ABA_trials + n_ABB_trials
"""

# Dimensionality of multiple-choice output
y_dim = 4
# Sequence length
seq_len = 9


# Method for calculating number of combinations
def n_comb(n, r):
    return int(math.factorial(n) / (math.factorial(r) * math.factorial(n - r)))


# Create subsampled dataset
def subsampled_dset(shapes, n_trials):
    seq = np.array([]).astype(np.int32)
    targ = np.array([]).astype(np.int32)
    while seq.shape[0] < n_trials:
        # Sample AAA trial
        np.random.shuffle(shapes)
        comb = shapes[:4]
        A1 = comb[0]
        A2 = comb[1]
        np.random.shuffle(comb)
        AAA_targ = np.where((comb == A2).astype(np.int32))[0][0]
        AAA_seq = [A1, A1, A1, A2, A2] + builtins.list(comb)
        if seq.shape[0] == 0:
            seq = np.expand_dims(np.array(AAA_seq), 0)
            targ = np.append(targ, AAA_targ)
        else:
            if not np.any(np.all(seq == np.tile(AAA_seq, [seq.shape[0], 1]), 1)):
                seq = np.append(seq, np.expand_dims(np.array(AAA_seq), 0), 0)
                targ = np.append(targ, AAA_targ)
            else:
                sample_again = True
                while sample_again:
                    # Sample another AAA trial
                    np.random.shuffle(shapes)
                    comb = shapes[:4]
                    A1 = comb[0]
                    A2 = comb[1]
                    np.random.shuffle(comb)
                    AAA_targ = np.where((comb == A2).astype(np.int32))[0][0]
                    AAA_seq = [A1, A1, A1, A2, A2] + builtins.list(comb)
                    if not np.any(np.all(seq == np.tile(AAA_seq, [seq.shape[0], 1]), 1)):
                        sample_again = False
                        seq = np.append(seq, np.expand_dims(np.array(AAA_seq), 0), 0)
                        targ = np.append(targ, AAA_targ)
        # Sample ABA trial
        np.random.shuffle(shapes)
        comb = shapes[:4]
        A1 = comb[0]
        B1 = comb[1]
        A2 = comb[2]
        B2 = comb[3]
        np.random.shuffle(comb)
        ABA_targ = np.where((comb == A2).astype(np.int32))[0][0]
        ABA_seq = [A1, B1, A1, A2, B2] + builtins.list(comb)
        if not np.any(np.all(seq == np.tile(ABA_seq, [seq.shape[0], 1]), 1)):
            seq = np.append(seq, np.expand_dims(np.array(ABA_seq), 0), 0)
            targ = np.append(targ, ABA_targ)
        else:
            sample_again = True
            while sample_again:
                np.random.shuffle(shapes)
                comb = shapes[:4]
                A1 = comb[0]
                B1 = comb[1]
                A2 = comb[2]
                B2 = comb[3]
                np.random.shuffle(comb)
                ABA_targ = np.where((comb == A2).astype(np.int32))[0][0]
                ABA_seq = [A1, B1, A1, A2, B2] + builtins.list(comb)
                if not np.any(np.all(seq == np.tile(ABA_seq, [seq.shape[0], 1]), 1)):
                    sample_again = False
                    seq = np.append(seq, np.expand_dims(np.array(ABA_seq), 0), 0)
                    targ = np.append(targ, ABA_targ)
        # Sample ABB trial
        np.random.shuffle(shapes)
        comb = shapes[:4]
        A1 = comb[0]
        B1 = comb[1]
        A2 = comb[2]
        B2 = comb[3]
        np.random.shuffle(comb)
        ABB_targ = np.where((comb == B2).astype(np.int32))[0][0]
        ABB_seq = [A1, B1, B1, A2, B2] + builtins.list(comb)
        if not np.any(np.all(seq == np.tile(ABB_seq, [seq.shape[0], 1]), 1)):
            seq = np.append(seq, np.expand_dims(np.array(ABB_seq), 0), 0)
            targ = np.append(targ, ABB_targ)
        else:
            sample_again = True
            while sample_again:
                np.random.shuffle(shapes)
                comb = shapes[:4]
                A1 = comb[0]
                B1 = comb[1]
                A2 = comb[2]
                B2 = comb[3]
                np.random.shuffle(comb)
                ABB_targ = np.where((comb == B2).astype(np.int32))[0][0]
                ABB_seq = [A1, B1, B1, A2, B2] + builtins.list(comb)
                if not np.any(np.all(seq == np.tile(ABB_seq, [seq.shape[0], 1]), 1)):
                    sample_again = False
                    seq = np.append(seq, np.expand_dims(np.array(ABB_seq), 0), 0)
                    targ = np.append(targ, ABB_targ)
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
    all_4shape_comb = builtins.list(combinations(shapes, 4))
    # All AAA trials
    all_AAA_seq = []
    all_AAA_targ = []
    for comb in all_4shape_comb:
        comb = builtins.list(comb)
        all_comb_perm = builtins.list(permutations(comb, 4))
        for A1 in comb:
            comb_minus_A1 = deepcopy(comb)
            comb_minus_A1.remove(A1)
            for A2 in comb_minus_A1:
                for choice_perm in all_comb_perm:
                    choice_perm = builtins.list(choice_perm)
                    AAA_targ = np.where((np.array(choice_perm) == A2).astype(np.int32))[0][0]
                    AAA_seq = [A1, A1, A1] + [A2, A2] + choice_perm
                    all_AAA_seq.append(AAA_seq)
                    all_AAA_targ.append(AAA_targ)
    # All ABA trials
    all_ABA_seq = []
    all_ABA_targ = []
    for comb in all_4shape_comb:
        comb = builtins.list(comb)
        all_comb_perm = builtins.list(permutations(comb, 4))
        for A1 in comb:
            comb_minus_A1 = deepcopy(comb)
            comb_minus_A1.remove(A1)
            for B1 in comb_minus_A1:
                comb_minus_B1 = deepcopy(comb_minus_A1)
                comb_minus_B1.remove(B1)
                for A2 in comb_minus_B1:
                    comb_minus_A2 = deepcopy(comb_minus_B1)
                    comb_minus_A2.remove(A2)
                    B2 = comb_minus_A2[0]
                    for choice_perm in all_comb_perm:
                        choice_perm = builtins.list(choice_perm)
                        ABA_targ = np.where((np.array(choice_perm) == A2).astype(np.int32))[0][0]
                        ABA_seq = [A1, B1, A1] + [A2, B2] + choice_perm
                        all_ABA_seq.append(ABA_seq)
                        all_ABA_targ.append(ABA_targ)
    # All ABB trials
    all_ABB_seq = []
    all_ABB_targ = []
    for comb in all_4shape_comb:
        comb = builtins.list(comb)
        all_comb_perm = builtins.list(permutations(comb, 4))
        for A1 in comb:
            comb_minus_A1 = deepcopy(comb)
            comb_minus_A1.remove(A1)
            for B1 in comb_minus_A1:
                comb_minus_B1 = deepcopy(comb_minus_A1)
                comb_minus_B1.remove(B1)
                for A2 in comb_minus_B1:
                    comb_minus_A2 = deepcopy(comb_minus_B1)
                    comb_minus_A2.remove(A2)
                    B2 = comb_minus_A2[0]
                    for choice_perm in all_comb_perm:
                        choice_perm = builtins.list(choice_perm)
                        ABB_targ = np.where((np.array(choice_perm) == B2).astype(np.int32))[0][0]
                        ABB_seq = [A1, B1, B1] + [A2, B2] + choice_perm
                        all_ABB_seq.append(ABB_seq)
                        all_ABB_targ.append(ABB_targ)
    # Duplicate AAA trials (so that trial types are balanced)
    max_trials_per_trial_type = np.max([len(all_AAA_seq), len(all_ABA_seq), len(all_ABB_seq)])
    all_AAA_seq_augmented = deepcopy(all_AAA_seq)
    all_AAA_targ_augmented = deepcopy(all_AAA_targ)
    for a in range(max_trials_per_trial_type - len(all_AAA_seq)):
        trial_ind = np.floor(np.random.rand() * len(all_AAA_seq)).astype(np.int32)
        all_AAA_seq_augmented.append(all_AAA_seq[trial_ind])
        all_AAA_targ_augmented.append(all_AAA_targ[trial_ind])
    all_AAA_seq = all_AAA_seq_augmented
    all_AAA_targ = all_AAA_targ_augmented
    # Combine same and different trials
    seq = all_AAA_seq + all_ABA_seq + all_ABB_seq
    targ = all_AAA_targ + all_ABA_targ + all_ABB_targ
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
        train_shapes, test_shapes
):
    log.info('n_shapes = ' + str(n_shapes) + '...')
    log.info('m_holdout = ' + str(m_holdout) + '...')
    # If m = 0, training and test sets are drawn from same set of shapes
    if m_holdout == 0:
        # Total number of possible trials
        shapes_avail = n_shapes
        n_AAA_trials = n_comb(shapes_avail, 4) * n_comb(4, 1) * n_comb(3, 1) * math.factorial(4)
        n_ABA_trials = n_comb(shapes_avail, 4) * n_comb(4, 1) * n_comb(3, 1) * n_comb(2, 1) * n_comb(1,
                                                                                                     1) * math.factorial(
            4)
        n_ABB_trials = n_comb(shapes_avail, 4) * n_comb(4, 1) * n_comb(3, 1) * n_comb(2, 1) * n_comb(1,
                                                                                                     1) * math.factorial(
            4)
        total_unique_trials = n_AAA_trials + n_ABA_trials + n_ABB_trials
        total_trials = np.max([n_AAA_trials, n_ABA_trials, n_ABB_trials]) * 3
        log.info('Total possible trials = ' + str(total_unique_trials) + '...')
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
        n_AAA_trials = n_comb(shapes_avail, 4) * n_comb(4, 1) * n_comb(3, 1) * math.factorial(4)
        n_ABA_trials = n_comb(shapes_avail, 4) * n_comb(4, 1) * n_comb(3, 1) * n_comb(2, 1) * n_comb(1,
                                                                                                     1) * math.factorial(
            4)
        n_ABB_trials = n_comb(shapes_avail, 4) * n_comb(4, 1) * n_comb(3, 1) * n_comb(2, 1) * n_comb(1,
                                                                                                     1) * math.factorial(
            4)
        total_unique_trials = n_AAA_trials + n_ABA_trials + n_ABB_trials
        total_trials = np.max([n_AAA_trials, n_ABA_trials, n_ABB_trials]) * 3
        log.info('Total possible training trials = ' + str(total_unique_trials) + '...')
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
        n_AAA_trials = n_comb(shapes_avail, 4) * n_comb(4, 1) * n_comb(3, 1) * math.factorial(4)
        n_ABA_trials = n_comb(shapes_avail, 4) * n_comb(4, 1) * n_comb(3, 1) * n_comb(2, 1) * n_comb(1,
                                                                                                     1) * math.factorial(
            4)
        n_ABB_trials = n_comb(shapes_avail, 4) * n_comb(4, 1) * n_comb(3, 1) * n_comb(2, 1) * n_comb(1,
                                                                                                     1) * math.factorial(
            4)
        total_unique_trials = n_AAA_trials + n_ABA_trials + n_ABB_trials
        total_trials = np.max([n_AAA_trials, n_ABA_trials, n_ABB_trials]) * 3
        log.info('Total possible test trials = ' + str(total_unique_trials) + '...')
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


class IdentityRules(Dataset):
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
        self.seq_len = 4

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
        for m in range(self.seq_len):
            full_img = 255 * np.ones((160, 160))
            transx, transy = np.random.randint(-5, 5, (2,))

            full_img = fill_image(self.all_imgs, seq_ind[0], full_img, [40 + transx, 40 + transy])
            transx, transy = np.random.randint(-5, 5, (2,))

            full_img = fill_image(self.all_imgs, seq_ind[1], full_img, [40 + transx, 88 + transy])
            transx, transy = np.random.randint(-5, 5, (2,))

            full_img = fill_image(self.all_imgs, seq_ind[2], full_img, [40 + transx, 136 + transy])
            transx, transy = np.random.randint(-5, 5, (2,))

            full_img = fill_image(self.all_imgs, seq_ind[3], full_img, [120 + transx, 40 + transy])
            transx, transy = np.random.randint(-5, 5, (2,))

            full_img = fill_image(self.all_imgs, seq_ind[4], full_img, [120 + transx, 88 + transy])
            transx, transy = np.random.randint(-5, 5, (2,))

            full_img = fill_image(self.all_imgs, seq_ind[5 + m], full_img, [120 + transx, 136 + transy])

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

