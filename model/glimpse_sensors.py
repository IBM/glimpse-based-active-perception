import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as VF

import cv2 as cv


class MultiScaleSensor(nn.Module):
    def __init__(self, scales, glimpse_size, padding_type='constant'):
        super().__init__()
        self.padding_type = padding_type
        self.glimpse_size = glimpse_size
        self.scales = scales

    @torch.no_grad()
    def initialize_image_scale_space(self, image):
        """extracts different scales from the image from which the patches will be cut out"""
        scale_to_img_dict = {0: image}
        for scale in self.scales:
            scale_to_img_dict[scale] = VF.gaussian_blur(image, kernel_size=scale**2 if scale % 2 == 1 else scale**2 + 1)
        return scale_to_img_dict
    
    def extract_patch(self, image, location, size, padding_type='constant'):
        image_size = image.shape[-1]
        batch_size = image.shape[0]

        # pad with zeros to avoid exceeding the range of the image by extracting the patch
        padded_image = F.pad(
            image, (size // 2, size // 2, size // 2, size // 2), mode=padding_type)

        # locations are in the range [-1, 1]
        # hack: since we pad the image we would need to add padding size ('size // 2') to offset the location
        #       which would be 'start = location - size // 2 + offset == location'
        start = (0.5 * ((location + 1.0) * (image_size-1))).long()
        end = start + size

        # loop through mini-batch and extract patches
        patch = []
        for b in range(batch_size):
            patch.append(padded_image[b, :, start[b, 1]: end[b, 1], start[b, 0]: end[b, 0]])
        return torch.stack(patch)

    def forward(self, scale_to_img_dict, location, **kwargs):
        """
        Parameters
        ----------
        scale_to_img_dict: is an output dict from 'extract_patch(...)' with keys corresponding to scales
        (scale '0' corresponds to the original image)
        location: xy location in range [-1, 1]
        Returns
        -------
        multiscale glimpse of shape [B, S, C, G, G] where G is glimpse size, C is number of channels and
        S is the number of scales incl. the original scale of the input image
        """
        assert isinstance(scale_to_img_dict, dict)

        with torch.no_grad():
            glimpse_windows = [self.extract_patch(scale_to_img_dict[0], location, self.glimpse_size, self.padding_type)]

            for scale in self.scales:
                patch = self.extract_patch(
                    scale_to_img_dict[scale], location, size=int(self.glimpse_size * scale),
                    padding_type=self.padding_type)

                patch = VF.resize(patch, self.glimpse_size)
                if patch.shape[1] == 1:
                    patch = patch / (patch.flatten(1).max(dim=-1).values[..., None, None, None] + 1e-6)

                glimpse_windows.append(patch)

            return torch.stack(glimpse_windows, dim=1)


class LogPolarSensor(nn.Module):
    def __init__(self, glimpse_size, radius, blur_kernel_size=7, log_polar=True, replication_padding=False):
        super().__init__()

        self.replication_padding = replication_padding
        self.glimpse_size = glimpse_size
        self.blur_kernel_size = blur_kernel_size
        self.radius = radius
        self.flag = cv.WARP_POLAR_LOG if log_polar else cv.WARP_POLAR_LINEAR
        self.flag += cv.WARP_FILL_OUTLIERS

    @torch.no_grad()
    def initialize_image_scale_space(self, image, **kwargs):
        # NOTE: blur is only needed to deal with binary SVRT images that have very thin shapes,  
        #       if no blur is applied on these images, OpenCV implementation of log-polar sampling 
        #       will not work properly. For more naturalstic images, blur is not necessary
        blurred_image = VF.gaussian_blur(image, kernel_size=[self.blur_kernel_size, self.blur_kernel_size])
        blurred_image /= blurred_image.flatten(1).max(-1).values[..., None, None, None]

        return torch.stack([
            image,
            blurred_image
        ], dim=1)

    def forward(self, scale_to_image_dict, location, **kwargs):
        """
        Parameters
        ----------
        scale_to_image_dict: is an output dict from 'extract_patch(...)' with keys corresponding to scales
        (scale '0' corresponds to the original image and '1' corresponds to its blurred version to more stable
        log-polar sampling)
        location: xy location in range [-1, 1]

        Returns
        -------
        multiscale glimpse of shape [B, C, G, G] where G is glimpse size, C is number of channels
        """
        batch_size = scale_to_image_dict.shape[0]
        blurred_image = scale_to_image_dict[:, 1]

        image_size = blurred_image.shape[-1]
        n_channels = blurred_image.shape[1]

        if self.replication_padding:
            blurred_image = F.pad(
                blurred_image,
                (self.radius, self.radius, self.radius, self.radius),
                mode='replicate'
            )

        blurred_image = blurred_image.permute(0, 2, 3, 1).detach().to('cpu').numpy()
        if n_channels == 1:
            blurred_image = blurred_image[..., 0]
        glimpses = []
        locs = (image_size * (location + 1) / 2).to('cpu')
        if self.replication_padding:
            locs += self.radius
        for b in range(batch_size):
            glimpse = cv.warpPolar(
                blurred_image[b],
                dsize=(self.glimpse_size, self.glimpse_size),
                center=(int(locs[b, 0].item()), int(locs[b, 1].item())),
                maxRadius=self.radius,
                flags=self.flag
            )
            if len(glimpse.shape) == 2:
                glimpse = glimpse[..., None]
            glimpses.append(torch.from_numpy(glimpse).permute(2, 0, 1))
        
        glimpses = torch.stack(glimpses, dim=0).to(location.device)[:, None]

        return glimpses

