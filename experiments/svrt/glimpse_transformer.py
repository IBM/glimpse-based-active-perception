from typing import Literal

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
from model.glimpse_sensors import MultiScaleSensor, LogPolarSensor
from model.glimpsing_process import SaliencyMapBasedGlimpsing
from model.error_neurons import ErrorNeurons
from utils import create_mlp
from trainer import Trainer
from datasets.svrt_full import SVRTFull
from model.downstream_architectures.transformer import TransformerDownstreamArchitecture

from definitions import DEVICE, DATA_PATH, LOGDIR_PATH_PREFIX


class Experiment(BaseExperiment):
    def __init__(
            self,

            task,
            n_samples,
            seed,
            sensor: Literal['logpolar', 'multiscale'],
    ):
        super().__init__(
            locals(),
            os.path.join(
                LOGDIR_PATH_PREFIX,
                "glimpse_models_logs/svrt/glimpse_transformer"))

        batch_size = 100
        cnn_size = 27
        timesteps = 15

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if DEVICE == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # this CNN is used to pre-process each glimpse depending on glimpse sensor type
        if sensor == 'multiscale':
            glimpse_size = 15
            scales = [4, 8]
            sensor = MultiScaleSensor(scales, glimpse_size)

            glimpse_embedder = nn.Sequential(
                nn.Flatten(0, 1),
                nn.Flatten(1, 2),
                nn.Conv2d(len(scales) + 1, cnn_size, 3),
                nn.ReLU(),
                nn.Conv2d(cnn_size, cnn_size, 3),
                nn.ReLU(),
                nn.Conv2d(cnn_size, cnn_size, 3),
                nn.ReLU(),
                nn.Conv2d(cnn_size, cnn_size, 3),
                nn.ReLU(),
                nn.Conv2d(cnn_size, cnn_size, 3),
                nn.ReLU(),
                nn.Conv2d(cnn_size, cnn_size, 3),
                nn.ReLU(),
                nn.Conv2d(cnn_size, cnn_size, 3),
                nn.ReLU(),
                transforms_v2.Lambda(lambda x: x.flatten(2).mean(-1)),
                nn.Unflatten(0, (batch_size, timesteps)),
            )
        else:
            scales = []
            glimpse_size = 21
            sensor = LogPolarSensor(glimpse_size, radius=48)
            glimpse_embedder = nn.Sequential(
                nn.Flatten(0, 1),
                nn.Flatten(1, 2),
                nn.Conv2d(len(scales) + 1, cnn_size, 5),
                nn.ReLU(),
                nn.Conv2d(cnn_size, cnn_size, 5),
                nn.ReLU(),
                nn.Conv2d(cnn_size, cnn_size, 5),
                nn.ReLU(),
                nn.Conv2d(cnn_size, cnn_size, 5),
                nn.ReLU(),
                nn.Conv2d(cnn_size, cnn_size, 3),
                nn.ReLU(),
                nn.Conv2d(cnn_size, cnn_size, 3),
                nn.ReLU(),
                transforms_v2.Lambda(lambda x: x.flatten(2).mean(-1)),
                nn.Unflatten(0, (batch_size, timesteps)),
            )

        # determine the dimensionality of pre-processed glimpses
        glimpse_hid_size = glimpse_embedder(
            torch.zeros(batch_size, timesteps, len(scales) + 1, 1, glimpse_size, glimpse_size)).shape[-1]
        glimpse_locs_mlp_sizes = [32, 64]   # sizes of MLP layers that will pre-process each glimpse location

        downstream_architecture = TransformerDownstreamArchitecture(
            glimpse_size=1,  # this class expects already pre-processed (by a CNN) glimpse
            glimpse_emd_dim=glimpse_hid_size,
            glimpse_locs_emb_dim=glimpse_locs_mlp_sizes[-1],
            heads=8, head_dim=64, mlp_dim=256, depth=12,
            readout_sizes=[64, 32, 2],
            timesteps=timesteps, n_glimpse_scales=len(scales) + 1,
            num_channels_in_glimpse=glimpse_hid_size,
            glimpse_embedder=glimpse_embedder,
            glimpse_locs_embedder=create_mlp(2, glimpse_locs_mlp_sizes),
            context_norm_glimpse_locs=True,
            context_norm_glimpses=True,
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
            normalize_glimpse_locs=True,
        )
        self.model.to(DEVICE)

        self.train_ds = DataLoader(
            SVRTFull(
                os.path.join(DATA_PATH, 'data'), True, task, num_samples=n_samples,
                transform=transforms_v2.Compose([
                    transforms_v2.RandomHorizontalFlip(),
                    transforms_v2.RandomVerticalFlip(),
                ])
            ),
            batch_size,
            shuffle=True)
        self.test_ds = DataLoader(
            SVRTFull(os.path.join(DATA_PATH, 'data'), False, task, num_samples=40000), batch_size)

        self.trainer = Trainer(self.log_dir, self.model, self.train_ds, self.test_ds, batch_size,
                               epochs=2000, eval_every=5,
                               device=DEVICE)


if __name__ == "__main__":
    exp = Experiment(
        task=1,
        n_samples=1000,
        seed=213,
        sensor='multiscale'
    )
    exp.run()
