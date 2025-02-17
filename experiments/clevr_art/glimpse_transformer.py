from typing import Literal
import os.path
import itertools
from collections import OrderedDict
import sys

import torch
from torch import nn

from torchvision.transforms import v2 as transforms_v2
from torchvision import transforms

import random
import numpy as np

from experiments.base_experiment import BaseExperiment
from model.assembled_final_model import GlimpseBasedModel
from model.glimpse_sensors import LogPolarSensor, MultiScaleSensor
from model.glimpsing_process import SaliencyMapBasedGlimpsing
from model.error_neurons import ErrorNeurons
from trainer import ArtDatasetCustomTrainer
from model.downstream_architectures.transformer import TransformerDownstreamArchitecture
from utils import create_mlp

from datasets.clevr_art import ClevrARTdataset


from definitions import DEVICE, LOGDIR_PATH_PREFIX, DATA_PATH


class Experiment(BaseExperiment):
    def __init__(
        self,

        seed,
        task: Literal['sd', 'rmts', 'dist3', 'id'],
        sensor: Literal['multiscale', 'logpolar']
    ):
        super().__init__(
            locals(),
            os.path.join(
                LOGDIR_PATH_PREFIX,
                "glimpse_models_logs/clevr_art/glimpse_transformer"))

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if DEVICE == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        batch_size = 32
        cnn_size = 64
    
        if task == 'sd' or task == 'rmts':
            timesteps = 20
        else:
            timesteps = 35

        if sensor == 'multiscale':
            scales = [4, 8]
            glimpse_size = 15
            sensor = MultiScaleSensor(scales, glimpse_size)
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
                nn.Unflatten(0, (-1, timesteps)),
            )
        else:
            scales = []
            glimpse_size = 21
            sensor = LogPolarSensor(glimpse_size, radius=48)
            glimpse_embedder = nn.Sequential(
                nn.Flatten(0, 1),
                nn.Flatten(1, 2),
                nn.Conv2d((len(scales) + 1) * 3, cnn_size, 5),
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
                nn.Unflatten(0, (-1, timesteps)),
            )
            
        glimpse_hid_size = glimpse_embedder(torch.zeros(batch_size, timesteps, len(scales)+1, 3, glimpse_size, glimpse_size)).shape[-1]
        glimpse_locs_mlp_sizes = [32, 64]
        
        downstream_architecture = TransformerDownstreamArchitecture(
            glimpse_size=1,  # this class expects already pre-processed (by a CNN) glimpse
            glimpse_emd_dim=glimpse_hid_size,
            glimpse_locs_emb_dim=glimpse_locs_mlp_sizes[-1],
            heads=8, head_dim=64, mlp_dim=256, depth=12,
            readout_sizes=[64, 32, 1],
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
                ior_mask_size=11, 
                error_neurons=ErrorNeurons(
                    center_size=5, neighbourhood_size=3, similarity_fn_name='mse', wta_surround=True,
                    normalize_patches=False, blurr_image=False, replication_padding=True
                ),
                soft_ior=False,
            ),
            n_glimpses=timesteps,
            downstream_architecture=downstream_architecture,
            normalize_glimpse_locs=True,
        )

        self.model.to(DEVICE)
        
        data_path = os.path.join(DATA_PATH, 'data/clevr_art')
        train_data = ClevrARTdataset(data_path, task, 'train', 128)
        test_data = ClevrARTdataset(data_path, task, 'test', 128)

        self.train_ds = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_ds = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        self.trainer = ArtDatasetCustomTrainer(self.log_dir, self.model, self.train_ds, self.test_ds, batch_size,
                               epochs=2000, eval_every=1, 
                               optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-4),
                               device=DEVICE)


if __name__ == "__main__":
    exp = Experiment(
        seed=1,
        task='rmts',
        sensor='logpolar',
    )
    exp.run()
