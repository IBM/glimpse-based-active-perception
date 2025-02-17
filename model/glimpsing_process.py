import numpy as np
import torch
import torch.nn as nn

from model.error_neurons import ErrorNeurons


class ImageLocationMasker(nn.Module):
    def __init__(self, image_size, mask_size=None, glimpse_size_to_mask_size_ratio=1):
        super().__init__()

        ranges = [np.linspace(-1., 1., num=res) for res in [image_size, image_size]]
        grid = np.meshgrid(*ranges, sparse=False, indexing="xy")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [image_size, image_size, -1])

        grid = grid.astype(np.float32)
        self.register_buffer('xy_grid', torch.from_numpy(grid).unsqueeze(0).permute(0, 3, 1, 2))

        self.mask_size = mask_size
        self.image_size = image_size
        if self.mask_size is not None:
            self.radius = ((self.mask_size ** 2) / (self.image_size ** 2)) * glimpse_size_to_mask_size_ratio / 2
        else:
            self.radius = None
        self.glimpse_size_to_mask_size_ratio = glimpse_size_to_mask_size_ratio

    def forward(self, xy_location, custom_mask_size=None):
        assert (xy_location <= 1.).all() and (xy_location >= -1.).all()
        if self.radius is None or custom_mask_size is not None:
            assert custom_mask_size is not None
            radius = ((custom_mask_size ** 2) / (self.image_size ** 2)) * self.glimpse_size_to_mask_size_ratio / 2
        else:
            radius = self.radius

        mask_distances = torch.square(self.xy_grid.flatten(2).swapaxes(1, -1) - xy_location.unsqueeze(1)).mean(-1)

        mask = torch.where(
            mask_distances <= radius,
            torch.ones_like(mask_distances, device=self.xy_grid.device),
            torch.zeros_like(mask_distances, device=self.xy_grid.device))
        mask = mask.reshape(-1, self.image_size, self.image_size)
        return mask


class SaliencyMapBasedGlimpsing(nn.Module):
    def __init__(
            self,
            image_size, ior_mask_size,
            error_neurons: ErrorNeurons,
            soft_ior=False,
            glimpse_size_to_mask_size_ratio=1,  # can be ignored
    ):
        super().__init__()

        self.soft_ior = soft_ior
        if image_size is not None:
            ranges = [np.linspace(-1., 1., num=res) for res in [image_size, image_size]]
            grid = np.meshgrid(*ranges, sparse=False, indexing="xy")
            grid = np.stack(grid, axis=-1)
            grid = np.reshape(grid, [image_size, image_size, -1])

            grid = grid.astype(np.float32)
            self.register_buffer('grid', torch.from_numpy(grid).unsqueeze(0).permute(0, 3, 1, 2))
        else:
            self.register_buffer('grid', None)

        self.image_size = image_size
        self.masker = ImageLocationMasker(self.image_size, ior_mask_size, glimpse_size_to_mask_size_ratio)
        self.error_neurons = error_neurons
        self.ior_mask_size = ior_mask_size

    @staticmethod
    def rbf(x, eps):
        return torch.exp(-torch.square(x * eps))

    def forward(
            self,
            saliency_map,
            ior_glimpse_hist_mask,
            current_location,
    ):
        saliency_map = saliency_map.unsqueeze(1)

        # block the current location in the saliency map using IoR
        mask = self.masker(current_location).unsqueeze(1)
        saliency_map = torch.where(mask.bool(), 0., saliency_map)

        if ior_glimpse_hist_mask is None:
            # initialize IoR mask 
            if self.soft_ior:
                ior_glimpse_hist_mask = torch.ones_like(saliency_map, device=saliency_map.device)
            else:
                ior_glimpse_hist_mask = torch.zeros_like(saliency_map, device=saliency_map.device)
        if self.soft_ior and torch.equal(ior_glimpse_hist_mask, torch.zeros_like(ior_glimpse_hist_mask)):
            # just for the case of soft IoR when an empty mask is provided (for soft IoR "empty mask" equals to the
            # mask of ones)
            ior_glimpse_hist_mask = torch.ones_like(ior_glimpse_hist_mask)

        if self.soft_ior:
            masked_saliency_map = ior_glimpse_hist_mask * saliency_map
        else:
            masked_saliency_map = torch.where(ior_glimpse_hist_mask > 0., torch.zeros_like(saliency_map), saliency_map)
        masked_xy_grid = torch.where(masked_saliency_map > 0., self.grid, -torch.inf)

        max_saliency, _ = masked_saliency_map.flatten(1).max(-1)
        masked_xy_grid = torch.where(
            masked_saliency_map >= max_saliency[..., None, None, None], masked_xy_grid, -torch.inf)

        distances = torch.square(masked_xy_grid.flatten(2).swapaxes(1, -1) - current_location.unsqueeze(1)).mean(-1)
        best_loc_idx = distances.argmin(-1)

        locations = torch.stack([best_loc_idx // self.image_size, best_loc_idx % self.image_size], -1)
        locations += 1

        # push locs into the range [-1, 1]
        locations = 2 * (locations.flip(-1) / self.image_size) - 1

        # if all salient points were already visited and only the points in the background are left, make the resulting
        # location look at the top left corner (at location [1,1])
        next_xy_location = torch.where(
            torch.eq(masked_saliency_map.flatten(1), torch.zeros_like(saliency_map).flatten(1)).all(-1)[..., None],
            torch.ones_like(locations), locations)

        if self.soft_ior:
            mask = self.rbf(
                (self.grid.permute(0, 2, 3, 1) - locations[:, None, None]).square().mean(-1),
                eps=self.ior_mask_size ** 2
            ).unsqueeze(1)
            ior_glimpse_hist_mask = ior_glimpse_hist_mask * (1 - mask)
        else:
            mask = self.masker(next_xy_location).unsqueeze(1)
            ior_glimpse_hist_mask = (ior_glimpse_hist_mask + mask).clip(0., 1.)

        return next_xy_location, ior_glimpse_hist_mask


