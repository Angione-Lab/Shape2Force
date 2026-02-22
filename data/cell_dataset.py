"""
Dataset and data loading for S2F training.
Expects folder structure: each subfolder has BF_001.tif (bright field), *_gray.jpg (heatmap), and optionally .txt (cell_area, sum_force).
"""
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from utils import config


def blur_force_map(force_map, ksize=25, sigma=10):
    if ksize % 2 == 0:
        ksize += 1
    if force_map.dim() == 3:
        force_map = force_map.unsqueeze(0)
    device = force_map.device
    force_map = force_map.cpu()
    blurred_maps = []
    for i in range(force_map.size(0)):
        force_np = force_map[i, 0].numpy().astype(np.float32)
        blurred = cv2.GaussianBlur(force_np, (ksize, ksize), sigmaX=sigma)
        blurred_maps.append(blurred)
    return torch.from_numpy(np.stack(blurred_maps)).to(device)


class ImageDataset(Dataset):
    def __init__(self, image_pairs, transform=None, channel_first=True,
                 blur_heatmap=False, threshold=0.0, return_metadata=False):
        self.image_pairs = image_pairs
        self.transform = transform
        self.channel_first = channel_first
        self.blur_heatmap = blur_heatmap
        self.threshold = threshold
        self.return_metadata = return_metadata

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        if self.return_metadata:
            bf_image, hm_image, numbers, metadata = self.image_pairs[idx]
        else:
            bf_image, hm_image, numbers = self.image_pairs[idx]
        if isinstance(numbers, tuple):
            cell_area, sum_force = numbers
        else:
            cell_area = 0
            sum_force = numbers

        image = torch.from_numpy(bf_image).float().unsqueeze(0)
        heatmap = torch.from_numpy(hm_image).float().unsqueeze(0)
        if self.transform:
            image, heatmap = self.transform(image, heatmap)
        cell_area = torch.tensor(cell_area, dtype=torch.float32)
        sum_force = torch.tensor(sum_force, dtype=torch.float32)
        heatmap[heatmap <= self.threshold] = 0
        if self.blur_heatmap:
            heatmap = blur_force_map(heatmap)
        if not self.channel_first:
            image = image.permute(2, 1, 0)
            heatmap = heatmap.permute(2, 1, 0)
        if self.return_metadata:
            return image, heatmap, cell_area, sum_force, metadata
        return image, heatmap, cell_area, sum_force


def load_image(filepath, target_size):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img.astype(np.float32)


def load_text_data(filepath):
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        cell_area_diff = float(lines[0].split(":")[1].strip()) * config.SCALE_FACTOR_AREA
        sum_force_diff = float(lines[1].split(":")[1].strip()) * config.SCALE_FACTOR_FORCE
        return (cell_area_diff, sum_force_diff)


def load_images_from_subfolders(root_folder, target_size, load_numerical_data=True,
                                load_force_sum=False, return_metadata=False, substrate=None):
    paired_images = []
    numerical_data = []
    metadata = []
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        bf_image_path = hm_image_path = txt_file_path = None
        for filename in os.listdir(subfolder_path):
            if filename.endswith("BF_001.tif"):
                bf_image_path = os.path.join(subfolder_path, filename)
            elif filename.endswith("_gray.jpg"):
                hm_image_path = os.path.join(subfolder_path, filename)
            elif filename.endswith(".txt"):
                txt_file_path = os.path.join(subfolder_path, filename)

        if return_metadata:
            if substrate is None:
                from utils.substrate_settings import list_substrates
                raise ValueError("substrate must be passed when return_metadata=True. Options: " +
                                 ", ".join(list_substrates()))
            metadata.append({'folder_name': subfolder, 'substrate': substrate, 'root_folder': root_folder})

        if load_numerical_data:
            if bf_image_path and hm_image_path and txt_file_path:
                paired_images.append((bf_image_path, hm_image_path))
                numerical_data.append(load_text_data(txt_file_path))
        elif load_force_sum:
            if bf_image_path and hm_image_path:
                paired_images.append((bf_image_path, hm_image_path))
                hm = load_image(hm_image_path, target_size)
                numerical_data.append((0, float(np.sum(hm)) * config.SCALE_FACTOR_FORCE))
        else:
            if bf_image_path and hm_image_path:
                paired_images.append((bf_image_path, hm_image_path))

    with ThreadPoolExecutor() as executor:
        bf_loaded = list(executor.map(lambda p: load_image(p[0], target_size), paired_images))
        hm_loaded = list(executor.map(lambda p: load_image(p[1], target_size), paired_images))
    if not numerical_data:
        numerical_data = [(0, 0)] * len(bf_loaded)
    if return_metadata:
        return list(zip(bf_loaded, hm_loaded, numerical_data, metadata))
    return list(zip(bf_loaded, hm_loaded, numerical_data))


def prepare_data(input_folder, batch_size=8, target_size=(1024, 1024), split_size=0.2,
                 use_augmentations=True, train_test_sep_folder=True, channel_first=True,
                 load_numerical_data=False, load_force_sum=False, blur_heatmap=False,
                 threshold=0.0, return_metadata=False, substrate=None):
    if load_numerical_data and load_force_sum:
        raise ValueError("load_numerical_data and load_force_sum cannot be True at the same time")

    if train_test_sep_folder:
        train_folder = os.path.join(input_folder, 'train')
        test_folder = os.path.join(input_folder, 'test')
        if not (os.path.exists(train_folder) and os.path.exists(test_folder)):
            raise ValueError(f"train/test folders not found in {input_folder}")
        train_pairs = load_images_from_subfolders(train_folder, target_size=target_size,
                                                  load_numerical_data=load_numerical_data,
                                                  load_force_sum=load_force_sum,
                                                  return_metadata=return_metadata, substrate=substrate)
        val_pairs = load_images_from_subfolders(test_folder, target_size=target_size,
                                                load_numerical_data=load_numerical_data,
                                                load_force_sum=load_force_sum,
                                                return_metadata=return_metadata, substrate=substrate)
    else:
        image_pairs = load_images_from_subfolders(input_folder, target_size=target_size,
                                                  load_numerical_data=load_numerical_data,
                                                  load_force_sum=load_force_sum,
                                                  return_metadata=return_metadata, substrate=substrate)
        train_pairs, val_pairs = train_test_split(image_pairs, test_size=split_size, random_state=42)

    train_transform = None
    if use_augmentations:
        from .augmentations import AdvancedAugmentations
        train_transform = AdvancedAugmentations(target_size)

    train_dataset = ImageDataset(train_pairs, transform=train_transform, channel_first=channel_first,
                                 blur_heatmap=blur_heatmap, threshold=threshold, return_metadata=return_metadata)
    train_dataset.name = os.path.basename(input_folder)
    val_dataset = ImageDataset(val_pairs, channel_first=channel_first,
                               blur_heatmap=blur_heatmap, threshold=threshold, return_metadata=return_metadata)
    val_dataset.name = os.path.basename(input_folder)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def load_folder_data(folder_path, substrate=None, img_size=1024, blur_heatmap=False,
                     batch_size=2, threshold=0.0, return_metadata=False):
    val_pairs = load_images_from_subfolders(folder_path, target_size=img_size,
                                            load_numerical_data=False, load_force_sum=False,
                                            return_metadata=return_metadata, substrate=substrate)
    val_dataset = ImageDataset(val_pairs, channel_first=True, blur_heatmap=blur_heatmap,
                               threshold=threshold, return_metadata=return_metadata)
    val_dataset.name = os.path.basename(folder_path)
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
