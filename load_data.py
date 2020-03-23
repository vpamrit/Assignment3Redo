import os
import random
import re
import torch
import torchvision
import nibabel as nib

from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from collections import namedtuple
from torchvision import transforms, utils
from PIL import Image

from os.path import join, isfile
from os import listdir

#constants
SAVE_IMAGES = False
TRAIN_DIR = 'img/'
IMG_PREFIX = 'img'
LABEL_DIR = 'label/'
LABEL_PREFIX = 'label'
EXT = '.nii.gz'
SPLEEN_VAL = 1

class Img:
    def __init__(self, img_name, img, label , idx, slice_num):
        self.img_name = img_name
        self.img = img
        self.label = label
        self.idx = idx
        self.slice_num = slice_num
        self.complete = False


class SpleenDatasetBuilder:
    def __init__(self, root_dir, img_range=(0,0)):
        self.root_dir = root_dir
        self.img_range = img_range


        subjects = []

        #check if there is a labels
        if self.root_dir[-1] != '/':
            self.root_dir += '/'

        self.is_labeled = os.path.isdir(self.root_dir + LABEL_DIR)

        self.files = [re.findall('[0-9]{4}', filename)[0] for filename in os.listdir(self.root_dir + TRAIN_DIR)]
        self.files = sorted(self.files, key = lambda f : int(f))

        # store all subjects in the list
        for img_num in range(img_range[0], img_range[1]+1):
            img_file = os.path.join(self.root_dir, TRAIN_DIR, IMG_PREFIX + self.files[img_num] + EXT)
            label_file = os.path.join(self.root_dir, LABEL_DIR, LABEL_PREFIX + self.files[self.img_num] + EXT)

            subjects.append(torchio.Subject([
                torchio.Image('t1', img_file, torchio.INTENSITY),
                torchio.Image('label', label_file, torchio.LABEL)
            ]))

            print(img_file)

        # Define transforms for data normalization and augmentation
        mtransforms = (
            ZNormalization(),
            RandomNoise(std_range=(0, 0.25)),
            RandomFlip(axes=(0,)),
        )

        self.subjects = torchio.ImagesDataset(subjects, transform=transforms.Compose(mtransforms))
        self.queue_dataset = torchio.Queue(
            subjects_dataset=self.subjects,
            max_length=500,
            samples_per_volume=675,
            sampler_class=torchio.data.GridSampler,
            patch_size=(3, 240, 240),
            num_workers=4,
            shuffle_subjects=False,
            shuffle_patches=True
        )

        print("Dataset details\n  Images: {}, 2D Slices: {}, Subslices {}, Padding-Margin: {}".format(self.img_range[1] - self.img_range[0] + 1, self.len, self.total_slices, self.padding))

        def get_dataset(self):
            return self.queue_dataset