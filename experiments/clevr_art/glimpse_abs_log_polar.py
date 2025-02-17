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
from model.downstream_architectures.abstractor import AbstractorDownstreamArchitecture
from utils import create_mlp

from definitions import DEVICE, LOGDIR_PATH_PREFIX, DATA_PATH
from datasets.clevr_art import ClevrARTdataset


class Experiment(BaseExperiment):
    def __init__(
            self,

            seed,
            task,

            external_log_dir=None,
    ):
        super().__init__(
            locals(),
            os.path.join(
                LOGDIR_PATH_PREFIX,
                "glimpse_models_logs/clevr_art/glimpse_abs_log_polar"),
            external_log_dir=external_log_dir
        )

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if DEVICE == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        cnn_size = 64
        mlp_dim = 512
        depth = 24
        heads = 8
        glimpse_size = 15
        timesteps = 45
        if task == 'rmts':
            timesteps = int(timesteps * 2 / 3)

        batch_size = 32

        scales = []
        sensor = LogPolarSensor(glimpse_size, radius=48)

        # define CNN to pre-process each glimpse
        n_conv_layers = glimpse_size // 2 - 1
        main_layers = list()
        for _ in range(n_conv_layers):
            main_layers.append(nn.Conv2d(cnn_size, cnn_size, 3))
            main_layers.append(nn.ReLU())
        glimpse_embedder = nn.Sequential(
            nn.Flatten(0, 1),
            nn.Flatten(1, 2),
            nn.Conv2d((len(scales) + 1) * 3, cnn_size, 3),
            nn.ReLU(),
            *main_layers,
            transforms_v2.Lambda(lambda x: x.flatten(2).mean(-1)),
            nn.Unflatten(0, (-1, timesteps)),
        )

        # determine the dimensionality of pre-processed glimpses
        glimpse_hid_size = glimpse_embedder(
            torch.zeros(batch_size, timesteps, (len(scales) + 1), 3, glimpse_size, glimpse_size)).shape[-1]
        glimpse_locs_mlp_sizes = [32, 64]   # sizes of MLP layers that will pre-process each glimpse location

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
                ior_mask_size=glimpse_size, glimpse_size_to_mask_size_ratio=1,
                error_neurons=ErrorNeurons(
                    center_size=5, neighbourhood_size=3,
                    similarity_fn_name='mse', wta_surround=True,
                    normalize_patches=False, blurr_image=False, replication_padding=True,
                ),
                soft_ior=True,
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

        # initialize our own OOD datasets on which no prior art model (such as OCRA or slot-abstractor) has been pretrained
        ood_test_sets = list()
        for custom_dataset_name in ['_pyramid_cone', '_pyramid_sphere']:
            ood_test_set = ClevrARTdataset(data_path, 'idrules', 'test', 128, custom_dataset_name=custom_dataset_name)
            ood_test_set = torch.utils.data.DataLoader(ood_test_set, batch_size=batch_size, shuffle=False)
            ood_test_sets.append(('ood_test' + custom_dataset_name, ood_test_set))

        self.trainer = ArtDatasetCustomTrainer(self.log_dir, self.model, self.train_ds, self.test_ds, batch_size,
                                               epochs=2000, eval_every=1,
                                               ext_eval_datasets=ood_test_sets,
                                               optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-4),
                                               device=DEVICE, binary_classification_regime=True)


if __name__ == "__main__":
    param_space = OrderedDict(
        seed=list(range(1, 6)),
        task=['rmts', 'idrules'],
    )
    param_configs = list(
        itertools.product(*list(param_space.values()))
    )

    hp_config_idx = int(sys.argv[1]) - 1
    params_values = param_configs[hp_config_idx]

    kwargs = dict()
    for i, name in enumerate(param_space.keys()):
        kwargs[name] = params_values[i]

    exp = Experiment(**kwargs)
    exp.run()
