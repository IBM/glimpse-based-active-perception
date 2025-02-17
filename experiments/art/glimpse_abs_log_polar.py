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
from model.glimpse_sensors import LogPolarSensor
from model.glimpsing_process import SaliencyMapBasedGlimpsing
from model.error_neurons import ErrorNeurons
from trainer import ArtDatasetCustomTrainer
from model.downstream_architectures.abstractor import AbstractorDownstreamArchitecture
from utils import create_mlp

from datasets.art.util import create_train_and_test_loaders
from datasets.art import identity_rules
from datasets.art import same_diff 
from datasets.art import RMTS 
from datasets.art import dist3

from definitions import DEVICE, LOGDIR_PATH_PREFIX, DATA_PATH


class Experiment(BaseExperiment):
    def __init__(
        self,

        seed,
        task: Literal['sd', 'rmts', 'dist3', 'id'],
    ):
        super().__init__(
            locals(),
            os.path.join(
                LOGDIR_PATH_PREFIX,
                "glimpse_models_logs/art/glimpse_abs_log_polar"))

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if DEVICE == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        batch_size = 32

        cnn_size = 64
        glimpse_size = 21
        depth=24
        heads=8
        mlp_dim=256
    
        if task == 'sd' or task == 'rmts':
            timesteps = 20
        else:
            timesteps = 35

        scales = []
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
            nn.Unflatten(0, (-1, timesteps)),
        )
        
        glimpse_hid_size = glimpse_embedder(torch.zeros(batch_size, timesteps, len(scales)+1, 1, glimpse_size, glimpse_size)).shape[-1]
        glimpse_locs_mlp_sizes = [32, 64]

        downstream_architecture = AbstractorDownstreamArchitecture(
            glimpse_hid_size, depth, heads, mlp_dim, timesteps, glimpse_loc_dim=glimpse_locs_mlp_sizes[-1],

            context_norm_glimpse_locs=True,
            context_norm_glimpses=True,

            glimpse_locs_embedder=create_mlp(2, glimpse_locs_mlp_sizes),
            glimpse_embedder=glimpse_embedder,
            num_classes=1,
        )

        self.model = GlimpseBasedModel(
            sensor,
            SaliencyMapBasedGlimpsing(
                image_size=128,
                ior_mask_size=11, glimpse_size_to_mask_size_ratio=1,
                error_neurons=ErrorNeurons(
                    center_size=5, neighbourhood_size=3, similarity_fn_name='mse', wta_surround=True,
                    normalize_patches=False, blurr_image=True, replication_padding=True
                ),
            ),
            n_glimpses=timesteps,
            downstream_architecture=downstream_architecture,
            normalize_glimpse_locs=True,
        )

        self.model.to(DEVICE)

        if task == 'id':
            dataset_cls = identity_rules.IdentityRules
            create_task_fn = identity_rules.create_task
        elif task == 'sd':
            dataset_cls = same_diff.SDdataset
            create_task_fn = same_diff.create_task
        elif task == 'dist3':
            dataset_cls = dist3.Dist3dataset
            create_task_fn = dist3.create_task
        elif task == 'rmts':
            dataset_cls = RMTS.RMTSdataset
            create_task_fn = RMTS.create_task
        else:
            raise NotImplementedError

        self.train_ds, self.test_ds = create_train_and_test_loaders(
            path=os.path.join(DATA_PATH, 'data/art/imgs'),
            img_size=128,
            batch_size=batch_size,
            n_shapes=100,
            m_holdout=95,
            train_set_size=10000,
            test_set_size=10000,
            train_gen_method='full_space',
            test_gen_method='subsample' if task == 'id' or task == 'rmts' else 'full_space',
            train_proportion=0.95,
            dataset_cls=dataset_cls,
            create_task_fn=create_task_fn,
            transformations=[transforms.Lambda(lambda X: 1 - X)],
        )
        
        self.trainer = ArtDatasetCustomTrainer(self.log_dir, self.model, self.train_ds, self.test_ds, batch_size,
                               epochs=2000, eval_every=10, 
                               optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-4),
                               device=DEVICE)


if __name__ == "__main__":
    exp = Experiment(
        seed=1,
        task='rmts',
    )
    exp.run()
