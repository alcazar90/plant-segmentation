#!/usr/bin/env python
# coding: utf-8
"""
    This benchmarking script loads a model checkpoint and evaluates it on the
    training, development, or testing split of the PlantDataset. In addition, the
    program can compute metrics over a set of alternative masks the dataset
    provides.

    Two things are needed to use the alternative masks: (i) These masks must be
    provided by the dataset (computed by another method). Mainly used for
    evaluation purposes and using the flag --alternative_masks to instantiate
    PlantDataset; (ii) a particular collate function to create the PyTorch
    DataLoaders that returns a triplet consisting of the input image, the ground
    truth mask, and the alternative mask.

    Example:
        python benchmarking.py --bs 2 --dataset_type cwt
        python benchmarking.py --bs 2 --dataset_type dead --model_checkpoint "ckpt/dead-single-segmentation.pth"

    Important: the dataset split are defined strictly by the random seed used to
    create the indices. It must be equal to the one used to train the model.
"""

import time
import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision

from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Compose, Grayscale

from dataset import PlantDataset, get_binary_target

from utils import (
    extract_tag_from_name,
    display_batch_masks,
    set_seed,
    )

from evaluation import (
    get_pred_label,
    compute_iou,
)

from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Parse arguments --------------------------------------------------------------
parser = argparse.ArgumentParser(description='Benchmarking script for the PlantDataset')
parser.add_argument("--bs", type=int, default=2, help="Batch size. Default=2")
parser.add_argument('--model_checkpoint', type=str, default='ckpt/cwt-single-segmentation.pth', help='path to the model checkpoint')
parser.add_argument("--dataset_type", type=str, default="cwt", help="Data type. Default=cwt")
parser.add_argument('--inventary_name', type=str, default='inventary.csv', help='name of the inventary csv file')
parser.add_argument('--seed', type=int, default=42313988, help='seed for reproducibility')
parser.add_argument('--show_n_images', type=int, default=None, help='number of images to show in the grid')
args = parser.parse_args()

if args.__dict__["bs"]  is not None:
    bs = args.__dict__["bs"]
if args.__dict__["model_checkpoint"]  is not None:
    model_checkpoint = args.__dict__["model_checkpoint"]
if args.__dict__["dataset_type"]  is not None:
    dataset_type = args.__dict__["dataset_type"]
if args.__dict__["inventary_name"]  is not None:
    inventary_name = args.__dict__["inventary_name"]
if args.__dict__["seed"]  is not None:
    seed_number = args.__dict__["seed"] 
if args.__dict__["show_n_images"]  is not None:
    show_n_images = args.__dict__["show_n_images"] 



# Load utility functions -------------------------------------------------------
def collate_fn(batch, target_fn=get_binary_target, alternative_masks=False):
    """
        Collate function to stack the masks as channels in the same tensor.
        get_target: function to get the target tensor from the masks and labels
            could be multi-labeling or binary.
        alternative_masks: if True, return the alternative masks for benchmarking
          in addition to the images, targets.
    """ 
    tfms = ToTensor()
    images = torch.cat([feature_extractor(example['image'], return_tensors='pt')['pixel_values'] for example in batch])
    if alternative_masks:
        # Resize to 512x512, then convert to grayscale and tensor. Finally, add 
        # a dimension for the channel (=1) with .unsqueeze(1) -> (B, 1, H, W)
        tfms_benchmark_masks = Compose([Grayscale(num_output_channels=1), ToTensor()])
        benchmark_masks = torch.cat([tfms_benchmark_masks(example['alternative_masks'].resize((512, 512))).unsqueeze(1) for example in batch])
    masks = [example['masks'] for example in batch]
    labels = [example['labels'] for example in batch]
    targets = torch.cat([target_fn(x[0], x[1], tfms, size=(512,512)) for x in zip(masks, labels)])

    # transformar a 1 cuando haya un entero distinto a 0 (semantic segmentation)
    targets = torch.where(targets > 0.0, 1.0, 0.0)
    if alternative_masks:
        if dataset_type == 'dead':
            # apply threshold to get a binary masks (dead alternative mask -> grayscale)
            benchmark_masks = torch.where(benchmark_masks > 0.6, 1.0, 0.0)
        return images, targets, benchmark_masks
    return images, targets


# Init dataset to evaluate -----------------------------------------------------
# Important: the dataset split are defined strictly by the random seed used in
# the file train.py to create the indices. 
if dataset_type == 'dead':
    label2id = {'dead':0, 'dead_cut': 1, 'noise': 2}
    dataset = PlantDataset('data', dataset_type, 'data_inventary.csv',
                           label2id=label2id, alternative_masks=True)
elif dataset_type == 'cwt':
    label2id = {'normal':0, 'normal_cut': 1, 'noise': 2}
    dataset = PlantDataset('data', dataset_type, 'data_inventary.csv',
                           label2id=label2id, alternative_masks=True)
else:
    AssertionError("Dataset type not found")


# DataLoaders ------------------------------------------------------------------
set_seed(seed_number)

val_pct = 0.15  # % of validation data

val_size= int(len(dataset) * val_pct)
train_size = len(dataset) - val_size
print('Train size:', train_size)
print('Validation size:', val_size)

# Split the dataset into training and validation sets
train_set, val_set = random_split(dataset, [train_size, val_size])

# Create data loaders for the training and validation sets
# Important: dataloaders don't shuffle the data, we want to track the observations
# with train_set.indices and val_set.indices to compute the metrics
train_loader = DataLoader(train_set, batch_size=bs, shuffle=False, num_workers=0,
                          collate_fn = lambda x: collate_fn(x, 
                                                            target_fn=get_binary_target,
                                                            alternative_masks=True))

val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=0,
                        collate_fn = lambda x: collate_fn(x, 
                                                          target_fn=get_binary_target,
                                                          alternative_masks=True))



# Model Config -----------------------------------------------------------------
model_name = "nvidia/mit-b0"
feature_extractor = SegformerImageProcessor.from_pretrained(model_name)

# Init Model -------------------------------------------------------------------
# 1 puede ser 'normal' o 'dead' dependiendo del dataset, pero solo consideramos
# entrenamiento binario en este archivo
stoi = {'detection': 1, 'non-detection': 0}
itos = {1: 'detection', 0: 'non-detection'}

model = SegformerForSemanticSegmentation.from_pretrained(
    model_name,
    id2label=stoi,
    label2id=itos,
)

# model to device
model=model.to(device)


# Load model checkpoint --------------------------------------------------------
ckpt_dict = torch.load(model_checkpoint)
model.load_state_dict(ckpt_dict['state_dict'])
model.eval()


# Benchmarking -----------------------------------------------------------------
# Compute metrics over the train set
_, train_labels, train_preds, _, train_alternative = get_pred_label(model, 
                                                                    train_loader, 
                                                                    device,
                                                                    alternative_masks=True)


train_iou_model = compute_iou(train_preds, train_labels)
train_iou_alternative = compute_iou(train_alternative, train_labels)

# Obtain the image name from the train_set.indices, dataloader must be shuffle=False
train_image_names = [train_set.dataset.images[idx] for idx in train_set.indices]

# Obtain the total number of masks per image and the number of masks associated
# to the detection tag (e.g. normal) in the train set

train_obs_masks = [[extract_tag_from_name(x) for x in dataset[idx]['masks']] for idx in train_set.indices]
train_num_masks = [len(x) for x in train_obs_masks]
train_normal_num_masks = [x.count('normal') for x in train_obs_masks]


train_df = pd.DataFrame({'image_name': train_image_names, 
                         'iou_model': train_iou_model, 
                         'iou_alternative': train_iou_alternative,
                         'num_masks': train_num_masks,
                         'normal_num_masks': train_normal_num_masks,
                         'split': 'train'})


print(f"Train pred mIoU: {train_iou_model.mean():.4f}")
print(f"Train alternative mIoU: {train_iou_alternative.mean():.4f}")

# Compute metrics over the validation set
val_images, val_labels, val_preds, _, val_alternative = get_pred_label(model, val_loader, device,
                                                              alternative_masks=True)
val_iou_model = compute_iou(val_preds, val_labels)
val_iou_alternative = compute_iou(val_alternative, val_labels)

# Obtain the image name from the val_set.indices, dataloader must be shuffle=False
val_image_names = [val_set.dataset.images[idx] for idx in val_set.indices]

# Obtain the total number of masks per image and the number of masks associated
# to the detection tag (e.g. normal) in the validation set

val_obs_masks = [[extract_tag_from_name(x) for x in dataset[idx]['masks']] for idx in val_set.indices]
val_num_masks = [len(x) for x in val_obs_masks]
val_normal_num_masks = [x.count('normal') for x in val_obs_masks]

val_df = pd.DataFrame({'image_name': val_image_names, 
                       'iou_model': val_iou_model, 
                       'iou_alternative': val_iou_alternative,
                       'num_masks': val_num_masks,
                       'normal_num_masks': val_normal_num_masks,
                       'split': 'val'})


print(f"Val pred mIoU: {val_iou_model.mean():.4f}")
print(f"Val alternative mIoU: {val_iou_alternative.mean():.4f}")

# Concatenate train and validation dataframes
results = pd.concat([train_df, val_df], ignore_index=True)
results['model_ckpt'] = model_checkpoint

# Save results in a csv file
results.to_csv(f'./results/benchmarking-{model_checkpoint.split("/")[-1].split(".")[0]}-{datetime.datetime.now().isoformat()[:-16]}.csv', index=False)
print(results)

# Save input images, prediction masks, and alternative masks in a grid for 
# validation observations
if show_n_images:
    # Hardcoded to show the last 5 images
    val_images = val_images[show_n_images:]
    val_preds = val_preds[show_n_images:]
    val_labels = val_labels[show_n_images:]
    val_alternative = val_alternative[show_n_images:]

# Inverse transform the images to display them
grid = torchvision.utils.make_grid(
    val_images,
    nrow=val_images.shape[0],
    padding=8,
    pad_value=90
)

grid_inv = grid * torch.tensor(feature_extractor.image_std).unsqueeze(1).unsqueeze(2) + torch.tensor(feature_extractor.image_mean).unsqueeze(1).unsqueeze(2)
fig = plt.figure(figsize=(16, 12))
plt.imshow(grid_inv.permute(1,2,0))
plt.axis('off')
#plt.title("Input images")
plt.savefig(f'./results/benchmarking-input-{model_checkpoint.split("/")[-1].split(".")[0]}-{datetime.datetime.now().isoformat()[:-16]}.png'
            ,dpi=320, bbox_inches='tight')

#display_batch_masks(val_preds.unsqueeze(1), caption="Predicted masks")
display_batch_masks(val_preds.unsqueeze(1), caption=None)
plt.savefig(f'./results/benchmarking-preds-{model_checkpoint.split("/")[-1].split(".")[0]}-{datetime.datetime.now().isoformat()[:-16]}.png'
            ,dpi=320, bbox_inches='tight')

#display_batch_masks(val_labels.unsqueeze(1), caption="Ground truth masks")
display_batch_masks(val_labels.unsqueeze(1), caption=None)
plt.savefig(f'./results/benchmarking-labels-{model_checkpoint.split("/")[-1].split(".")[0]}-{datetime.datetime.now().isoformat()[:-16]}.png'
            ,dpi=320, bbox_inches='tight')

#display_batch_masks(val_alternative.unsqueeze(1), caption="Alternative method")
display_batch_masks(val_alternative.unsqueeze(1), caption=None)
plt.savefig(f'./results/benchmarking-alternative-{model_checkpoint.split("/")[-1].split(".")[0]}-{datetime.datetime.now().isoformat()[:-16]}.png'
            ,dpi=320, bbox_inches='tight')
