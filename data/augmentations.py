import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch
from scipy.ndimage import gaussian_filter, map_coordinates


class AdvancedAugmentations:
    def __init__(self, target_size=(1024, 1024)):
        self.target_size = target_size

    def __call__(self, image, heatmap):
        image = TF.to_pil_image(image)
        heatmap = TF.to_pil_image(heatmap)

        if np.random.rand() > 0.5:
            image = TF.hflip(image)
            heatmap = TF.hflip(heatmap)
        if np.random.rand() > 0.5:
            image = TF.vflip(image)
            heatmap = TF.vflip(heatmap)

        if np.random.rand() > 0.5:
            angle = np.random.uniform(-45, 45)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            heatmap = TF.rotate(heatmap, angle, interpolation=TF.InterpolationMode.BILINEAR)

        if np.random.rand() > 0.5:
            width, height = image.size
            crop_size = int(min(width, height) * np.random.uniform(0.8, 1.0))
            i, j, h, w = T.RandomCrop.get_params(image, (crop_size, crop_size))
            image = TF.crop(image, i, j, h, w)
            heatmap = TF.crop(heatmap, i, j, h, w)
            image = TF.resize(image, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)
            heatmap = TF.resize(heatmap, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)

        if np.random.rand() > 0.5:
            image, heatmap = self.random_affine(image, heatmap)

        if not isinstance(image, torch.Tensor):
            image = TF.to_tensor(image)
        if not isinstance(heatmap, torch.Tensor):
            heatmap = TF.to_tensor(heatmap)

        if np.random.rand() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness_factor)
        if np.random.rand() > 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            image = TF.adjust_contrast(image, contrast_factor)

        if np.random.rand() > 0.5:
            noise_level = np.random.uniform(0.01, 0.05)
            noise = torch.randn_like(image) * noise_level
            image = torch.clamp(image + noise, 0, 1)

        if np.random.rand() > 0.5:
            image, heatmap = self.elastic_transform(image, heatmap)

        return image, heatmap

    def random_affine(self, image, heatmap):
        degrees = [-10.0, 10.0]
        translate = [0.05, 0.05]
        scale = [0.95, 1.05]
        shear = [-5.0, 5.0]
        params = T.RandomAffine.get_params(degrees, translate, scale, shear, image.size)
        angle, translate, scale, shear = params
        translate = list(translate)
        shear = list(shear)
        image = TF.affine(image, angle, translate, scale, shear, interpolation=TF.InterpolationMode.BILINEAR)
        heatmap = TF.affine(heatmap, angle, translate, scale, shear, interpolation=TF.InterpolationMode.BILINEAR)
        return image, heatmap

    def elastic_transform(self, image, heatmap, alpha=50, sigma=4):
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).numpy()
            heatmap_np = heatmap.permute(1, 2, 0).numpy()
        else:
            image_np = np.asarray(image)
            heatmap_np = np.asarray(heatmap)
            if image_np.ndim == 2:
                image_np = image_np[:, :, np.newaxis]
            if heatmap_np.ndim == 2:
                heatmap_np = heatmap_np[:, :, np.newaxis]

        shape = image_np.shape[:2]
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        image_transformed = np.zeros_like(image_np)
        heatmap_transformed = np.zeros_like(heatmap_np)
        for i in range(image_np.shape[2]):
            image_transformed[..., i] = map_coordinates(image_np[..., i], indices, order=1).reshape(shape)
        for i in range(heatmap_np.shape[2]):
            heatmap_transformed[..., i] = map_coordinates(heatmap_np[..., i], indices, order=1).reshape(shape)

        return torch.from_numpy(image_transformed).float().permute(2, 0, 1), \
               torch.from_numpy(heatmap_transformed).float().permute(2, 0, 1)
