import os.path
import itertools
from collections import OrderedDict
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision.transforms import v2 as transforms_v2

import random
import numpy as np

from experiments.base_experiment import BaseExperiment
from model.assembled_final_model import GlimpseBasedModel
from model.glimpse_sensors import MultiScaleSensor
from model.glimpsing_process import SaliencyMapBasedGlimpsing
from model.error_neurons import ErrorNeurons
from utils import create_mlp
from trainer import Trainer
from datasets.svrt_full import SVRTFull
from model.downstream_architectures.abstractor import AbstractorDownstreamArchitecture

from definitions import DEVICE, DATA_PATH, LOGDIR_PATH_PREFIX


class Experiment(BaseExperiment):
    def __init__(
            self,

            task,
            n_samples,
            seed,
    ):
        super().__init__(
            locals(),
            os.path.join(
                LOGDIR_PATH_PREFIX,
                f"glimpse_models_logs/svrt/glimpse_abs_multi_scales"))

        glimpse_size = 15
        timesteps = 15
        depth = 24
        glimpse_locs_mlp_sizes = [32, 64]  # sizes of MLP layers that will pre-process each glimpse location
        cnn_size = 27
        mlp_dim = 256

        batch_size = 100

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if DEVICE == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        scales = [4, 8]
        sensor = MultiScaleSensor(scales, glimpse_size)

        # this CNN is used to pre-process each glimpse
        # for SVRT, using depth-convs to process each scale of multi-scale glimpses separately
        # seems to work slightly better
        glimpse_embedder = nn.Sequential(
            nn.Flatten(0, 1),
            nn.Flatten(1, 2),
            nn.Conv2d(len(scales) + 1, cnn_size, 3, groups=len(scales) + 1),
            nn.ReLU(),
            nn.Conv2d(cnn_size, cnn_size, 3, groups=len(scales) + 1),
            nn.ReLU(),
            nn.Conv2d(cnn_size, cnn_size, 3, groups=len(scales) + 1),
            nn.ReLU(),
            nn.Conv2d(cnn_size, cnn_size, 3, groups=len(scales) + 1),
            nn.ReLU(),
            nn.Conv2d(cnn_size, cnn_size, 3, groups=len(scales) + 1),
            nn.ReLU(),
            nn.Conv2d(cnn_size, cnn_size, 3, groups=len(scales) + 1),
            nn.ReLU(),
            nn.Conv2d(cnn_size, cnn_size, 3, groups=len(scales) + 1),
            nn.ReLU(),
            transforms_v2.Lambda(lambda x: x.reshape(x.shape[0], -1, len(scales) + 1, *x.shape[-2:]).sum(2)),
            transforms_v2.Lambda(lambda x: x.flatten(2).mean(-1)),
            nn.Flatten(1),
            nn.Unflatten(0, (batch_size, timesteps)),
        )

        # determine the dimensionality of pre-processed glimpses
        glimpse_hid_size = glimpse_embedder(
            torch.zeros(batch_size, timesteps, len(scales) + 1, 1, glimpse_size, glimpse_size)).shape[-1]

        downstream_architecture = AbstractorDownstreamArchitecture(
            glimpse_hid_size, depth, 8, mlp_dim, timesteps, glimpse_loc_dim=glimpse_locs_mlp_sizes[-1],

            context_norm_glimpse_locs=True,
            context_norm_glimpses=True,

            glimpse_locs_embedder=create_mlp(2, glimpse_locs_mlp_sizes),
            glimpse_embedder=glimpse_embedder
        )

        self.model = GlimpseBasedModel(
            sensor,
            SaliencyMapBasedGlimpsing(
                image_size=128,
                ior_mask_size=glimpse_size, glimpse_size_to_mask_size_ratio=1,
                error_neurons=ErrorNeurons(
                    center_size=5, neighbourhood_size=3, similarity_fn_name='mse', wta_surround=True,
                    normalize_patches=False, blurr_image=True,
                ),
            ),
            n_glimpses=timesteps,
            downstream_architecture=downstream_architecture,
        )

        self.model.to(DEVICE)

        self.train_ds = DataLoader(
            SVRTFull(
                os.path.join(DATA_PATH, 'data'),
                True, task, num_samples=n_samples,
                transform=transforms_v2.Compose([
                    transforms_v2.RandomHorizontalFlip(),
                    transforms_v2.RandomVerticalFlip()])
            ),
            batch_size,
            shuffle=True)
        self.test_ds = DataLoader(
            SVRTFull(os.path.join(DATA_PATH, 'data'), False, task, num_samples=40000),
            batch_size)

        self.trainer = Trainer(self.log_dir, self.model, self.train_ds, self.test_ds, batch_size,
                               epochs=2000, eval_every=5,
                               save_best=False,
                               device=DEVICE)


if __name__ == "__main__":
    exp = Experiment(
        task=1,  # number of SVRT task, has to be an int from range [1, 23]
        n_samples=1000,  # size of the training dataset
        seed=213,  # seeds run in our work include [123, 321, 213]
    )
    exp.run()
