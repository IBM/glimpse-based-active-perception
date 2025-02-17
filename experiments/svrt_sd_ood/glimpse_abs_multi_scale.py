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
from trainer import Trainer
from datasets.svrt_full import SVRTFull, OOD_SVRTFull
from model.downstream_architectures.abstractor import AbstractorDownstreamArchitecture
from utils import create_mlp

from definitions import DEVICE, DATA_PATH, LOGDIR_PATH_PREFIX


class Experiment(BaseExperiment):
    def __init__(
            self,
            seed,
    ):
        super().__init__(
            locals(),
            os.path.join(
                LOGDIR_PATH_PREFIX,
                "glimpse_models_logs/svrt_sd_ood/glimpse_abs_multi_scale"))

        batch_size = 100

        glimpse_size = 15
        glimpse_locs_mlp_sizes = [32, 64]
        cnn_size = 27
        timesteps = 15
        depth = 24
        mlp_dim = 256
        heads = 8

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
        glimpse_embedder = nn.Sequential(
            nn.Flatten(0, 1),
            nn.Flatten(1, 2),
            nn.Conv2d((len(scales) + 1) * 3, cnn_size, 3),
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

        # determine the dimensionality of pre-processed glimpses
        glimpse_hid_size = glimpse_embedder(
            torch.zeros(batch_size, timesteps, len(scales) + 1, 3, glimpse_size, glimpse_size)).shape[-1]
        glimpse_locs_mlp_sizes = [32, 64]

        downstream_architecture = AbstractorDownstreamArchitecture(
            glimpse_hid_size, depth, heads, mlp_dim, timesteps, glimpse_loc_dim=glimpse_locs_mlp_sizes[-1],

            context_norm_glimpse_locs=True,
            context_norm_glimpses=True,

            glimpse_locs_embedder=create_mlp(2, glimpse_locs_mlp_sizes),
            glimpse_embedder=glimpse_embedder,
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
                os.path.join(DATA_PATH, 'data'), True, 1, num_samples=28000, rgb_format=True
            ),
            batch_size,
            shuffle=True)
        self.test_ds = DataLoader(
            SVRTFull(os.path.join(DATA_PATH, 'data'), False, 1, num_samples=2000, rgb_format=True),
            batch_size)

        dataset_names = [
            'random_color',
            'irregular',
            'regular',
            'open',
            'wider_line',
            'scrambled',
            'filled',
            'lines',
            'arrows',
            'rectangles',
            'straight_lines',
            'connected_squares',
            'connected_circles'
        ]
        self.ood_test_sets = list()
        for ds_name in dataset_names:
            ds = DataLoader(
                OOD_SVRTFull(ds_name, root=os.path.join(DATA_PATH, 'data'), rgb_format=True), batch_size)
            self.ood_test_sets.append((ds_name, ds))

        self.trainer = Trainer(self.log_dir, self.model, self.train_ds, self.test_ds, batch_size,
                               epochs=2000, eval_every=50,
                               ext_eval_datasets=self.ood_test_sets,
                               eval_after_epoch=199,
                               optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-4),
                               device=DEVICE)


if __name__ == "__main__":
    exp = Experiment(seed=1)  # overall, 10 seeds from range(1, 11) were tried
    exp.run()
