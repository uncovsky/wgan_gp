# Description:
# This file should contain custom dataset class. The class should subclass the torch.utils.data.Dataset.

import albumentations as A

# from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt

from pathlib import Path
import random

from skimage import io

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2


def load_validate_images(image_path):
    """
        Reads the rgb/gray directories, comparing the number of images in
        each.

        Assumes the directory structure is as follows:
        image_path/
            img_rgb/
                    <image1>.png
                    <image2>.png
                    ...
            img_gray/
                    <image1>.png
                    <image2>.png
                    ...

        Also checks that every rgb image has a corresponding gray image (same
        names).

        Returns a pd dataframe with rows corresponding to paths to rgb/gray
        datapoints.
    """
    rgb_path = Path(image_path, 'img_rgb')
    gray_path = Path(image_path, 'img_gray')

    rgb_images = list(rgb_path.glob('*/*.png'))
    gs_images = list(gray_path.glob('*/*.png'))

    assert len(rgb_images) == len(gs_images), \
        "Number of images in img_rgb and img_gray do not match"

    samples = []

    for img in rgb_images:
        filename = img.name
        dirname = img.parent.stem
        gray_match = Path(gray_path, dirname, filename)

        title = "_".join(img.stem.split('_')[:-1])

        assert gray_match.exists(), \
            f"Missing grayscale image {filename} in {dirname}"

        samples.append({
            'title': title,
            'rgb_path': img,
            'gray_path': gray_match
        })

    return samples

def train_test_split(data, test_size=0.2, seed=42):
    """
        Returns a train/test split of dataset of given proportions.
        The split is done randomly, unless `seed` is fixed.
    """
    random.seed(seed)
    random.shuffle(data)

    split_idx = int(len(data) * (1 - test_size))
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    return train_data, test_data


class PlacesDataset(Dataset):

    def __init__(self, data):
        self.data = data

        # No preprocessing except conversion to float32
        self.preprocess = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        self.preprocess_rgb = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datapoint = self.data[idx]

        rgb_img = io.imread(str(datapoint['rgb_path']))
        gray_img = io.imread(str(datapoint['gray_path']))
        title = datapoint['title']

        transformed_rgb = self.preprocess_rgb(rgb_img)
        transformed_gray = self.preprocess(gray_img)

        return transformed_gray, transformed_rgb, title
