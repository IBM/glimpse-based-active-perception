import sys
import random
import math
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import builtins
from itertools import combinations, permutations

# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True
# Logging utility
from datasets.art.util import log

"""
Combinatorics:
For n objects, there are c = nC3 possible combinations of 3 objects.
For each of c combinations, there are p = 3*2*1 permutations.
There are mp = (3*2*1)**2 meta-permutations (allowing for the same permuation to appear in both rows).
Number of total possible trials is mp * c = ((3*2*1)**2) * nC3).
For training set, n will actually be (n - m).
For test set, n will actually be (n - (n-m)).
"""

# Dimensionality of multiple-choice output
y_dim = 4
# Sequence length
seq_len = 9


# Method for calculating number of combinations
def n_comb(n, r):
    return int(math.factorial(n) / (math.factorial(r) * math.factorial(n - r)))


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
        n_row_comb = n_comb(shapes_avail, 3)
        n_metaperm = (3 * 2 * 1) ** 2
        total_trials = n_metaperm * n_row_comb
        log.info('Total possible trials = ' + str(total_trials) + '...')
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
    else:
        # Total number of possible training trials
        shapes_avail = n_shapes - m_holdout
        n_row_comb = n_comb(shapes_avail, 3)
        n_metaperm = (3 * 2 * 1) ** 2
        total_trials = n_metaperm * n_row_comb
        log.info('Total possible training trials = ' + str(total_trials) + '...')
        if train_set_size > total_trials:
            log.info('Desired training set size (' + str(
                train_set_size) + ') is larger than total number of possible training trials for this task (' + str(
                total_trials) + ')...')
            log.info('Changing training set size to ' + str(total_trials) + '...')
            train_set_size = total_trials
        else:
            log.info('Training set size = ' + str(train_set_size) + '...')
        # Total number of possible training trials
        shapes_avail = n_shapes - (n_shapes - m_holdout)
        n_row_comb = n_comb(shapes_avail, 3)
        n_metaperm = (3 * 2 * 1) ** 2
        total_trials = n_metaperm * n_row_comb
        log.info('Total possible test trials = ' + str(total_trials) + '...')
        if test_set_size > total_trials:
            log.info('Desired test set size (' + str(
                test_set_size) + ') is larger than total number of possible test trials for this task (' + str(
                total_trials) + ')...')
            log.info('Changing test set size to ' + str(total_trials) + '...')
            test_set_size = total_trials
        else:
            log.info('Test set size = ' + str(test_set_size) + '...')

    # Generate complete matrix problems
    # If m = 0, training and test sets are drawn from same set of shapes
    if m_holdout == 0:
        # Create all possible combinations
        all_row_comb = builtins.list(combinations(train_shapes, 3))
        # Create all possible permutations for each combination
        all_comb_perm = builtins.list(permutations(range(3)))
        # Create all trials
        all_trials = []
        for comb in all_row_comb:
            for perm1 in all_comb_perm:
                for perm2 in all_comb_perm:
                    all_trials.append(np.array([np.array(comb)[np.array(perm1)], np.array(comb)[np.array(perm2)]]))
        random.shuffle(all_trials)
        all_trials = np.array(all_trials)
        # Split trials for train and test sets
        trials_train = all_trials[:train_set_size, :, :]
        trials_test = all_trials[train_set_size:(train_set_size + test_set_size), :, :]
    # Otherwise, training and test sets are completely disjoint (in terms of the shapes that are used), and can be generated separately
    else:
        # Training trials
        # Create all possible combinations
        all_row_comb = builtins.list(combinations(train_shapes, 3))
        # Create all possible permutations for each combination
        all_comb_perm = builtins.list(permutations(range(3)))
        # Create all trials
        all_trials = []
        for comb in all_row_comb:
            for perm1 in all_comb_perm:
                for perm2 in all_comb_perm:
                    all_trials.append(np.array([np.array(comb)[np.array(perm1)], np.array(comb)[np.array(perm2)]]))
        random.shuffle(all_trials)
        all_trials = np.array(all_trials)
        # Split trials for train and test sets
        trials_train = all_trials[:train_set_size, :, :]
        # Test trials
        # Create all possible combinations
        all_row_comb = builtins.list(combinations(test_shapes, 3))
        # Create all possible permutations for each combination
        all_comb_perm = builtins.list(permutations(range(3)))
        # Create all trials
        all_trials = []
        for comb in all_row_comb:
            for perm1 in all_comb_perm:
                for perm2 in all_comb_perm:
                    all_trials.append(np.array([np.array(comb)[np.array(perm1)], np.array(comb)[np.array(perm2)]]))
        random.shuffle(all_trials)
        all_trials = np.array(all_trials)
        # Split trials for train and test sets
        trials_test = all_trials[:test_set_size, :, :]

    # Generate multiple-choice options
    # Training set
    train_answer_choices = []
    for t in range(trials_train.shape[0]):
        problem_shapes = trials_train[t, 0, :]
        other_shapes = train_shapes[
            np.all(np.not_equal(np.expand_dims(problem_shapes, 1), np.expand_dims(train_shapes, 0)), 0)]
        if other_shapes.shape[0] == 0:
            log.info(
                'Training set does not have enough objects for 4 multiple-choice options, limiting to 3 options...')
            all_choices = problem_shapes
        else:
            np.random.shuffle(other_shapes)
            other_choice = other_shapes[0]
            all_choices = np.append(problem_shapes, other_choice)
        np.random.shuffle(all_choices)
        train_answer_choices.append(all_choices)
    # Test set
    test_answer_choices = []
    for t in range(trials_test.shape[0]):
        problem_shapes = trials_test[t, 0, :]
        other_shapes = test_shapes[
            np.all(np.not_equal(np.expand_dims(problem_shapes, 1), np.expand_dims(test_shapes, 0)), 0)]
        if other_shapes.shape[0] == 0:
            log.info('Test set does not have enough objects for 4 multiple-choice options, limiting to 3 options...')
            all_choices = problem_shapes
        else:
            np.random.shuffle(other_shapes)
            other_choice = other_shapes[0]
            all_choices = np.append(problem_shapes, other_choice)
        np.random.shuffle(all_choices)
        test_answer_choices.append(all_choices)

    # Create different versions of sequence and targets for multiple-choice and predictive versions of task
    # Training set
    train_MC_seq = []
    train_MC_targ = []
    # val_MC_seq = []
    # val_MC_targ = []
    for t in range(0, trials_train.shape[0]):
        pre_MC = trials_train[t, :, :].flatten()[:-1]
        MC_seq = np.concatenate([pre_MC, train_answer_choices[t]])
        img_targ_id = trials_train[t, -1, -1]
        MC_targ = np.where(train_answer_choices[t] == img_targ_id)[0][0]
        train_MC_seq.append(MC_seq)
        train_MC_targ.append(MC_targ)

    # for t in range(int(0.9*trials_train.shape[0]),trials_train.shape[0]):
    # 	pre_MC = trials_train[t,:,:].flatten()[:-1]
    # 	MC_seq = np.concatenate([pre_MC, train_answer_choices[t]])
    # 	img_targ_id = trials_train[t,-1,-1]
    # 	MC_targ = np.where(train_answer_choices[t] == img_targ_id)[0][0]
    # 	val_MC_seq.append(MC_seq)
    # 	val_MC_targ.append(MC_targ)
    # Test set
    test_MC_seq = []
    test_MC_targ = []
    for t in range(trials_test.shape[0]):
        pre_MC = trials_test[t, :, :].flatten()[:-1]
        MC_seq = np.concatenate([pre_MC, test_answer_choices[t]])
        img_targ_id = trials_test[t, -1, -1]
        MC_targ = np.where(test_answer_choices[t] == img_targ_id)[0][0]
        test_MC_seq.append(MC_seq)
        test_MC_targ.append(MC_targ)

    # Create training and test sets
    # np.savez('/scratch/gpfs/smondal/slot_attention_reasoning/train_heldout95.npz',seq_ind = np.array(train_MC_seq), y = np.array(train_MC_targ))
    # np.savez('/scratch/gpfs/smondal/slot_attention_reasoning/test_heldout95.npz',seq_ind = np.array(test_MC_seq), y = np.array(test_MC_targ))

    train_set = {'seq_ind': np.array(train_MC_seq), 'y': np.array(train_MC_targ)}
    # val_set = {'seq_ind': np.array(val_MC_seq), 'y': np.array(val_MC_targ)}

    test_set = {'seq_ind': np.array(test_MC_seq), 'y': np.array(test_MC_targ)}

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


class Dist3dataset(Dataset):
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
