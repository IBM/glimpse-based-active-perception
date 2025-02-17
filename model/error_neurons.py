from typing import Literal

import torchvision.transforms.functional as VF
import torch
import torch.nn as nn
import torch.nn.functional as F


def gather_related_locations(locations_grid, loc_inds, invalid=0.):
    _, N, D = locations_grid.shape  # N = H*W, D - dim(feature dimension)
    B, T, S = loc_inds.shape  # T = H*W - num "glimpses", each location/node has its own glimpse window

    if loc_inds.max() == N:
        # special case: indices where idx == N is assigned zero, i.e. for each location at the border of the grid
        #               the features of neighbouring locations are just set to zero
        locations_grid = torch.cat(
            [locations_grid, invalid * torch.ones([B, 1, D], device=locations_grid.device)], dim=1)

    indices = loc_inds.reshape(B, T * S).unsqueeze(-1).expand(-1, -1, D)
    output = torch.gather(locations_grid, 1, indices).reshape([B, T, S, D])

    return output


def generate_local_indices(img_size, K, padding='constant', dilation=1, device='cpu'):
    if isinstance(img_size, tuple) or isinstance(img_size, list):
        H, W = img_size
    else:
        H, W = img_size, img_size

    indice_maps = torch.arange(H * W, device=device).reshape([1, 1, H, W]).float()

    # symmetric padding
    assert K % 2 == 1  
    half_K = int((K - 1) / 2) * dilation

    assert padding in ['reflection', 'constant'], "unsupported padding mode"
    if padding == 'reflection':
        pad_fn = torch.nn.ReflectionPad2d(half_K)
    else:
        pad_fn = torch.nn.ConstantPad2d(half_K, H * W)
    # padding is needed to provide neighbours for nodes at the borders

    indice_maps = pad_fn(indice_maps)
    local_inds = F.unfold(indice_maps, kernel_size=K, stride=1, dilation=dilation)  # [B, C * K * k, H, W]
    local_inds = local_inds.permute(0, 2, 1)
    return local_inds


def extract_pixel_based_center_surround_saliency_map(
        image,
        center_size: int,
        neighbourhood_size: int,
        dilation=1,
        similarity_fn_name: Literal['cos', 'mse'] = 'mse',
        wta_surround=False,
        blurr_image=False,
        normalize_patches=True,
        blurr_size=None,
        saliency_threshold=None,
        replication_padding=False,
        resize=None
):
    """
    Parameters
    ----------
    image: [B, C, H, W]
    center_size: size of the central receptive field to be compared with its surroundings
    neighbourhood_size: size of the surroundings from which the receptive fields of size 'center_size' will be
        extracted and compared with the central one
    dilation: can be ignored
    similarity_fn_name: similarity function to compare the central receptive field with the surrounding one
    wta_surround: if True, the 'min(...)' aggregation operator is used, otherwise the averaging is used
    blurr_image: might help to make saliency detection on SVRT more stable, because objects are too thin
    blurr_size:
    normalize_patches: can be ignored
    saliency_threshold: can be ignored
    replication_padding: works better for RGB images
    resize: can be ignored

    Returns
    -------

    """
    if blurr_image:
        if blurr_size is None:
            blurr_size = 7
        image = VF.gaussian_blur(image, [blurr_size, blurr_size])
        image /= image.flatten(2).max(-1).values[..., None, None]
    if resize is not None:
        image = VF.resize(image, [resize, resize])
    if replication_padding:
        bu_preds = F.unfold(nn.ReplicationPad2d(center_size//2)(image), center_size, stride=1).permute(0, 2, 1)
    else:
        bu_preds = F.unfold(image, center_size, stride=1, padding=center_size // 2).permute(0, 2, 1)

    if normalize_patches:
        bu_preds = F.layer_norm(bu_preds.reshape(*bu_preds.shape[:2], image.shape[1], -1), [center_size**2]).reshape(*bu_preds.shape) + 1.
        bu_preds += 1.

    B, H, W = image.shape[0], image.shape[-2], image.shape[-1]

    # neighbour_inds.shape is B, H*W, S = local_window_size^2;
    # one can see 'sample_inds' as a grid where each node contains indices of neighbouring locations
    # with which the internal representation will be compared
    neighbour_inds = generate_local_indices(
        img_size=[H, W], K=neighbourhood_size, device=image.device, dilation=dilation).expand(
        B, -1, -1).long()  # [B, H*W, S], S = N^2, N - local_window_size

    # node_inds: [B, H*W, S]
    node_inds = torch.arange(H * W, device=image.device).reshape([1, H * W, 1]).expand(B, -1, neighbourhood_size ** 2)

    # cut out the central node idx in each glimpse window
    center_node_idx = (neighbourhood_size ** 2) // 2
    neighbour_inds = neighbour_inds[..., torch.arange(neighbourhood_size ** 2).long() != center_node_idx]
    node_inds = node_inds[..., :-1]

    gathered_nodes = gather_related_locations(bu_preds, node_inds)
    gathered_surrounding_nodes = gather_related_locations(bu_preds, neighbour_inds)

    if similarity_fn_name == 'cos':
        cos_sim = F.cosine_similarity(gathered_nodes, gathered_surrounding_nodes, dim=-1)
        similarity_distance = 1 - ((1 + cos_sim) / 2)
        # similarity_distance = ((1 + torch.sum(gathered_nodes * gathered_surrounding_nodes, dim=-1)) / 2)
    elif similarity_fn_name == 'mse':
        similarity_distance = torch.sum(torch.square(gathered_nodes - gathered_surrounding_nodes), dim=-1)
    else:
        raise NotImplementedError

    if wta_surround:
        similarity_distance = torch.where(neighbour_inds == H*W, torch.inf, similarity_distance).min(-1).values.reshape(-1, H, W)
    else:
        invalid_border_locs_mask = torch.where(neighbour_inds == H * W, 0., 1.)
        similarity_distance = (similarity_distance * invalid_border_locs_mask).sum(-1) / invalid_border_locs_mask.sum(-1)
        similarity_distance = similarity_distance.reshape(-1, H, W)

    saliency_map = similarity_distance

    if saliency_threshold is not None:
        saliency_map = torch.where(saliency_map >= saliency_threshold, saliency_map, 0.)

    return saliency_map


class ErrorNeurons(nn.Module):
    """
    This is just a class-based wrapper for the function 'extract_pixel_based_center_surround_saliency_map(...)'
    """
    def __init__(
            self,
            center_size,
            neighbourhood_size,
            dilation=1,
            similarity_fn_name: Literal['cos', 'mse'] = 'cos',
            wta_surround=False,
            blurr_image=False,
            normalize_patches=True,
            blurr_size=None,
            saliency_threshold=None,
            replication_padding=False,
            resize=None,
    ):
        
        super().__init__()

        self.resize = resize
        self.replication_padding = replication_padding
        self.blurr_size = blurr_size
        self.normalize_patches = normalize_patches
        self.blurr_image = blurr_image
        self.wta_surround = wta_surround
        self.similarity_fn_name = similarity_fn_name
        self.dilation = dilation
        self.neighbourhood_size = neighbourhood_size
        self.center_size = center_size
        self.saliency_threshold = saliency_threshold

    def forward(self, image):
        return extract_pixel_based_center_surround_saliency_map(
            image, self.center_size, self.neighbourhood_size, self.dilation, self.similarity_fn_name,
            self.wta_surround,
            self.blurr_image, self.normalize_patches, self.blurr_size,
            self.saliency_threshold,
            self.replication_padding,
            self.resize
        )

