"""
    File: dataset.py

    Implement the PyTorch Dataset to navigate through the images and masks in
    the folder structure of the dataset.

    TODO:
        - Improve docstring for PlantDataset
        - Implement a fail-safe in case the mask_idx is not found in the mask directory
        - in the csv file the name of the column has a type in it, it should be 'id original img' instead of 'id orginal img'
"""

import os
import re 
import numpy as np
import pandas as pd

import torch

from PIL import Image
from torch.utils.data import Dataset
from utils import extract_ids_from_name, extract_tag_from_name

class PlantDataset(Dataset):
    """
        Dataset class to navigate through the images and masks in the folder.
        The folder structure is the following:

        root
        ├── folder
        │   ├── original
        │   │   ├── 0_0.jpg

        │   ├── original_labeled
        │   │   ├── 0
        │   │   │   ├── task-0-annotation-0-0-0.jpg
        │   │   │   ├── task-0-annotation-0-0-1.jpg
        │   │   │   ├── task-0-annotation-0-0-2.jpg

        
        label2id should be a dictionary with the class names and its corresponding
        integer representation. For example:
            label2id = {'normal': 0,
                        'normal_cut': 1,    
                        'noise': 2,}
        It's important that the labels name provide in label2id are equal to the
        labels used in the name structure of each mask to avoid key error probelms.
    """
    def __init__(self, 
                 root, 
                 folder, 
                 inventary_name, 
                 transform=None,
                 label2id=None,
                 alternative_masks=False):
        """ Assuming inventary name is within the root folder """
        self.root = root
        self.folder = folder
        self.transform = transform
        self._label2id = label2id
        self.alternative_masks = alternative_masks

        assert label2id is not None, 'label2id parameter must be provided (e.g. {"normal": 0, "normal_cut": 1, "noise": 2}))'
        
        # create label-to-id and id-to-label dictionaries with the
        # class names and its corresponding integer representation
        if self._label2id is None:
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

        # alternative_masks is a boolean flag to indicate if we want to add alternative masks
        # provided by the dataset (computed by other method). Mainly used for evaluation purposes.
        if self.alternative_masks: 
            self.benchmark_masks = []
            alternative_masks_folder = os.listdir(os.path.join(self.root, self.folder, 'original_labeled_imageJ_mask'))
            alternative_masks_folder.sort(key=lambda x: (int(re.findall(r'\d+', x.split('_')[0])[0]), 
                                                int(re.findall(r'\d+', x.split('_')[2])[0])))

        for i, img in enumerate(files_in_folder):
            self.images.append(img)

            if self.alternative_masks:
                self.benchmark_masks.append(alternative_masks_folder[i])

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

        if self.alternative_masks:
            alternative_mask_path = os.path.join(self.root, self.folder, 'original_labeled_imageJ_mask', self.benchmark_masks[idx])
            return {
                'image': image,
                'masks': masks_path,
                'labels': labels,
                'alternative_masks': Image.open(alternative_mask_path)
            }
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
        pattern = ''.join([s + '-|' if idx < len(self._label2id)-1 else s for idx, s in enumerate(self._label2id.keys())])
        get_label = lambda x: re.findall(pattern, x)[0].replace('-', '')
        out = [get_label(m) for img in self.masks for m in img]
        return np.unique(out, return_counts=True)


def get_target(masks, labels, tfms, size=(250, 250), num_classes=4):
    """ 
        Recibe una lista de máscaras y una lista de etiquetas, retorna
        un tensor de dimensiones (n_classes, height, widht). En cada clase,
        cualquier entero distinto a 0 representa a uns instancia particular
        de la clase. El entero 0 se reserva para background/no-clase.
    """
    # creamos tensor para almacenar máscaras por clase en cada canal (dim 1)
    # NOTA: la última es ausencia o ninguna detecctión 
    out = torch.zeros((1, num_classes, size[0], size[1]))

    # si no hay máscaras, retornamos el tensor
    if len(masks) == 0:
        return out

    # crear un entero que represente cada instancia de las mascaras asociadas
    # a la observación. No siguen un orden necesario tipo todas las de la clase 0
    # parten al principio, se encuentran según el orden de los labels. El entero
    # 0 se reserva como background/no-clase
    instance_idxs = torch.tensor([l+1 for l in range(len(labels))], dtype=torch.long)

    # iteramos sobre cada clase para procesar las máscaras asociadas a estas,
    # agregar un identificador de instancias y colapsar en una sola matriz
    # Supuesto: no hay clase sobrelapadas. Si hay, se debe hacer un proceso
    # adicional para ver que entero se asigna al pixel correspondiente
    for l in list(set(labels)):
        x = torch.cat([torch.where(tfms(Image.open(m).resize(size)) > 0.0, 1.0, 0.0) * instance_idxs[i] for i, m in enumerate(masks) if labels[i] == l])
        out[0, l, :, :] = x.sum(dim=0)

    return out


def get_binary_target(masks, labels, tfms, size=(250, 250)):
    """ 
        Recibe una lista de máscaras y una lista de etiquetas, retorna
        un tensor de dimensiones (1, height, width). La clase target_id
        se identificar con el entero 1, el resto es representado por
        el entero 0 (es decir background o noise).

        Supuesto: se espera que la clase con la señal a detectar siempre sea la
        representada por el entero 0 en el diccionario dataset._label2id. Por ejemplo,
        si queremos detectar la clase "normal" (ver dataset._label2id), entonces:
            - dataset._label2id -> {'normal': 0, 'normal_cut': 1, 'noise': 2}
    """
    target_id = 0
    out = torch.zeros((1, 1, size[0], size[1]))

    # si no hay máscaras o la clase a detectar no tiene máscara, retornamos el tensor
    if len(masks) == 0 or target_id not in (labels):
        return out
    
    target_masks_idx = np.where(np.array(labels) == target_id)[0]

    target_masks=[]
    for idx in target_masks_idx:
        target_masks.append(torch.where(tfms(Image.open(masks[idx]).resize(size)) > 0.0, 1.0, 0.0))
    out[0,0,:,:] = torch.where(torch.cat(target_masks).sum(dim=0) > 0.0, 1.0, 0.0)
    return out
