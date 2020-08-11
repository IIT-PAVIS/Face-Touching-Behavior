#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
	IIT : Istituto italiano di tecnologia

    Pattern Analysis and Computer Vision (PAVIS) research line

    Usage Example:
		$ python3

    LICENSE:
	This project is licensed under the terms of the MIT license.
	This project incorporates material from the projects listed below (collectively, "Third Party Code").
	This Third Party Code is licensed to you under their original license terms.
	We reserves all other rights not expressly granted, whether by implication, estoppel or otherwise.
	The software can be freely used for any non-commercial applications.
"""

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os

import torch
from torch.utils.data import Dataset

# Defining the pyTorch custom dataset
class FaceTouchDataset(Dataset):
    def __init__(self, dictionary, transform=None, log_enabled=False):
        train_dict = {
            "imgs": dictionary['imgs'],
            "labels": dictionary['labels']
        }
        self.img_filenames = dictionary['imgs']
        self.labels = dictionary['labels']
        self.transform = transform

        self.log_enabled = log_enabled

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.img_filenames[idx]

        if os.path.isfile(filename):
            img = Image.open(filename)
            if not img.mode == 'RGB':
                img = img.convert('RGB')
        else:
            img = None

        # If corrupted image, generating an empty one with fake label
        if img is None or len(img.getbands()) < 3:
            if self.log_enabled:
                if img is None:
                    print("Corrupted image! - {} - Path: {} - Img is None!".format(idx, filename))
                else:
                    print("Corrupted image! - {} - Path: {} - len(img.getbands())".format(
                        idx, self.img_filenames[idx], len(img.getbands())))

            img = Image.fromarray(np.zeros([300, 300, 3], dtype=np.uint8))
            label = 0 * int(self.labels[idx])
        else:
            label = int(self.labels[idx])

        if self.transform:
            img = self.transform(img)

        return (img, label, filename)

    def __len__(self):
        return len(self.img_filenames)
