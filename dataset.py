"""
    File: dataset.py

    Implement the PyTorch Dataset to navigate through the images and masks in
    the folder structure of the dataset.

    TODO:
        - Capture the label associated of each mask from the name
        - Implement a fail-safe in case the mask_idx is not found in the mask directory
        - in the csv file the name of the column has a type in it, it should be 'id original img' instead of 'id orginal img'
"""

import os
import re 
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from utils import extract_ids_from_name, extract_tag_from_name

class PlantDataset(Dataset):

    def __init__(self, root, folder, inventary_name, transform=None):
        """ Assuming inventary name is within the root folder """
        self.root = root
        self.folder = folder
        self.transform = transform
        
        # create label-to-id and id-to-label dictionaries with the
        # class names and its corresponding integer representation
        self._label2id = {'normal': 0,
                      'normal_cut': 1,
                      'noise': 2,}
        self._id2label = {idx: ch for ch, idx in self._label2id.items()}

        # read inventary csv file and store it in a pandas dataframe
        self.idx_table = pd.read_csv(os.path.join(self.root, self.folder, inventary_name))

        # self.masks is a nested list in which in each position we have 1 or more
        # masks associated to the image in the same position in the images list
        # same for self.labels, a nested list but with the labels (integers)
        # of each mask
        self.images = []
        self.masks = []
        self.labels = []            
        self.mask_idx_folder = []   # populate when iterate trough the images 

        # iterate over files in root + folder path + 'original'
        files_in_folder = os.listdir(os.path.join(self.root, self.folder, 'original'))

        # ensures that the order of the files follows the original image id and then the split id
        files_in_folder.sort(key=lambda x: (int(re.findall(r'\d+', x.split('_')[0])[0]), 
                                            int(re.findall(r'\d+', x.split('_')[1])[0])))

        for img in files_in_folder:
            self.images.append(img)

            # Get the id of the image's mask folder from the inventary (self.idx_table)
            # TODO: in the csv file the name of the column has a type in it, 
            # it should be 'id original img' instead of 'id orginal img'
            id_original_img, id_after_split = extract_ids_from_name(img)

            mask_idx = self.idx_table[(self.idx_table['id orginal img'] == id_original_img) & 
                                  (self.idx_table['id after split'] == id_after_split)]['id']._values[0]

            self.mask_idx_folder.append(str(mask_idx))
            # TODO: implement a fail-safe in case the mask_idx is not found in the mask directory
            masks = [x for x in os.listdir(os.path.join(self.root, self.folder, 'original_labeled', str(mask_idx)))]
            self.masks.append(masks)

            labels = [self._label2id[extract_tag_from_name(m)] for m in masks]
            self.labels.append(labels)

    def __len__(self):
        """ Return the length of the dataset (# of images) """
        return len(self.images)
    
    def __getitem__(self, idx):
        """ Return the image and the masks associated to it """
        image_path = os.path.join(self.root, self.folder, 'original', self.images[idx])
        image = Image.open(image_path)
        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        masks_path = [str(os.path.join(self.root, self.folder, 'original_labeled', self.mask_idx_folder[idx], m)) for m in self.masks[idx]]
        labels = self.labels[idx]

        return {
                'image': image, 
                'masks': masks_path, 
                'labels': labels
        }
    
    def get_number_of_masks(self):
        """ Return the number of masks per image """
        return [len(m) for m in self.masks]
    
    def get_masks_per_labels(self):
        """ Return the number of masks per label """
        get_label = lambda x: re.findall(r"normal-|noise|normal_cut+", x)[0].replace('-', '')
        out = [get_label(m) for img in self.masks for m in img]
        return np.unique(out, return_counts=True)
