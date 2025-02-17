import torch
import torch.nn as nn

from model.glimpsing_process import SaliencyMapBasedGlimpsing
from model.glimpse_sensors import LogPolarSensor, MultiScaleSensor
from model.downstream_architectures.abstractor import AbstractorDownstreamArchitecture
from model.downstream_architectures.transformer import TransformerDownstreamArchitecture
import utils


class GlimpseBasedModel(nn.Module):
    def __init__(
            self,
            sensor: LogPolarSensor or MultiScaleSensor,
            glimpse_action: SaliencyMapBasedGlimpsing,
            n_glimpses: int,
            downstream_architecture: AbstractorDownstreamArchitecture or TransformerDownstreamArchitecture,
            normalize_glimpse_locs=False,
            grayscale_to_rgb=False,
    ):
        super().__init__()

        self.readout = downstream_architecture
        self.glimpse_action = glimpse_action
        self.sensor = sensor
        self.n_glimpses = n_glimpses
        self.normalize_glimpse_locs = normalize_glimpse_locs
        self.grayscale_to_rgb = grayscale_to_rgb

    def forward(self, img, **kwargs):
        scale_to_img_dict = self.sensor.initialize_image_scale_space(img)

        if self.grayscale_to_rgb:
            img = img.repeat(1, 3, 1, 1)

        saliency_map = self.glimpse_action.error_neurons(img)

        loc_hist = list()
        glimpse_hist = list()

        ior_mask_hist = torch.zeros(saliency_map.shape[0], 1, *saliency_map.shape[-2:], device=img.device)
        locations = torch.ones(img.shape[0], 2, device=img.device)

        for t in range(self.n_glimpses):
            locations, ior_mask_hist = self.glimpse_action.forward(
                saliency_map, ior_mask_hist, locations)

            glimpse = self.sensor(scale_to_img_dict, locations, as_dict=False)

            loc_hist.append(locations)
            glimpse_hist.append(glimpse)
        
        glimpse_hist = torch.stack(glimpse_hist, dim=1)
        loc_hist = torch.stack(loc_hist, dim=1)

        if self.normalize_glimpse_locs: 
            loc_hist = utils.standardize_glimpse_locs(loc_hist)

        out = self.readout(glimpses=glimpse_hist, glimpse_locs=loc_hist)
   
        return out
