#!/usr/bin/env python
# coding: utf-8
"""
    File: train.py

    Train a Single Segmentation Model using SegFormer from HuggingFace Transformers
    and the Plant dataset.
"""

import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor

from tqdm import tqdm

from dataset import PlantDataset, get_binary_target

from utils import (
    upscale_logits, # quizas deberian ir al modulo del modelo?
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

# Argparser --------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=2, help="Batch size. Default=2")
parser.add_argument("--model_name", type=str, default="nvidia/mit-b0", help="Model Name. Default=nvidia/mit-b0")
parser.add_argument("--dataset_type", type=str, default="cwt", help="Data type. Default=cwt")
parser.add_argument("--epochs", type=int, default=2, help="Number of Epochs to train. Default=2")
parser.add_argument("--eval_steps", type=int, default=1, help="Number of steps to get evaluation. Default=1")
parser.add_argument("--lr", type=float, default=1e-3, help="Use folloup question. Default=1e-3")
parser.add_argument("--rep", type=int, default=1, help="Number of repetitions. Default=1")
#parser.add_argument("--logtodb", type=str2bool, help="Log Run To DB. Default=True")
args = parser.parse_args()


if args.__dict__["bs"]  is not None:
    bs = args.__dict__["bs"]
if args.__dict__["model_name"]  is not None:
    model_name = args.__dict__["model_name"]
if args.__dict__["dataset_type"]  is not None:
    dataset_type = args.__dict__["dataset_type"]
if args.__dict__["epochs"]  is not None:
    epochs = args.__dict__["epochs"]
if args.__dict__["eval_steps"]  is not None:
    eval_steps = args.__dict__["eval_steps"]
if args.__dict__["lr"]  is not None:
    lr = args.__dict__["lr"]  
if args.__dict__["rep"]  is not None:
    rep = args.__dict__["rep"] 


# Load utility functions -------------------------------------------------------
def collate_fn(batch, target_fn=get_binary_target):
    """
        Collate function to stack the masks as channels in the same tensor.
        get_target: function to get the target tensor from the masks and labels
            could be multi-labeling or binary.
    """ 
    # Acá se pueden agregar todas las transformaciones adicionales de preproceso
    # que se requieran para las máscaras. Lo única esencial es pasar una PIL.Image
    # a tensor
    tfms = ToTensor()
    images = torch.cat([feature_extractor(example['image'], return_tensors='pt')['pixel_values'] for example in batch])
    masks = [example['masks'] for example in batch]
    labels = [example['labels'] for example in batch]
    targets = torch.cat([target_fn(x[0], x[1], tfms, size=(512,512)) for x in zip(masks, labels)])

    # transformar a 1 cuando haya un entero distinto a 0 (semantic segmentation)
    targets = torch.where(targets > 0.0, 1.0, 0.0)
    return images, targets

# Load Dataset -----------------------------------------------------------------
if dataset_type == 'dead':
    label2id = {'dead':0, 'dead_cut': 1, 'noise': 2}
    dataset = PlantDataset('data', dataset_type, 'data_inventary.csv', label2id=label2id)
elif dataset_type == 'cwt':
    label2id = {'normal':0, 'normal_cut': 1, 'noise': 2}
    dataset = PlantDataset('data', dataset_type, 'data_inventary.csv', label2id=label2id)
else:
    AssertionError("Dataset type not found")

# Model Config -----------------------------------------------------------------
model_name = "nvidia/mit-b0"
feature_extractor = SegformerImageProcessor.from_pretrained(model_name)

# Save train config ------------------------------------------------------------
configuration = {
    'bs':bs,
    'model_name': model_name,
    'dataset_type': dataset_type,
    'epochs':epochs,
    'eval_steps':eval_steps,
    'lr':lr
}

short_configuration = ''
for configValue in configuration:
    short_configuration = short_configuration + str(configuration[configValue])

print(f"Training config: {configuration}")

# DataLoaders ------------------------------------------------------------------
seed_number = 42313988
set_seed(seed_number)
configuration['seed_number'] = seed_number


val_pct = 0.15  # % of validation data
configuration['val_pct'] = val_pct

val_size= int(len(dataset) * val_pct)
train_size = len(dataset) - val_size
print('Train size:', train_size)
print('Validation size:', val_size)

# Split the dataset into training and validation sets
train_set, val_set = random_split(dataset, [train_size, val_size])
print(train_set)

# Create data loaders for the training and validation sets
train_loader = DataLoader(train_set, batch_size=bs, 
                          shuffle=True, num_workers=0,
                          collate_fn = lambda x: collate_fn(x, target_fn=get_binary_target))

val_loader = DataLoader(val_set, batch_size=bs, 
                        shuffle=False, num_workers=0,
                        collate_fn = lambda x: collate_fn(x, target_fn=get_binary_target))


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

# Optimizer and loss function --------------------------------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


# Training loop ----------------------------------------------------------------
# compute running time
start_time = time.time()

lossi = np.zeros(epochs)
val_lossi = []
miou = []
steps = 0
best_miou = -np.inf
for idx in tqdm(range(epochs)):
    cur_loss = 0.0
    for xb, yb in train_loader:
        model.train()
        xb = xb.to(device)
        yb = yb.to(device)

        # Perform a forward pass
        logits = model(xb)["logits"]

        # Clean the gradients  
        optimizer.zero_grad()

        # Compute the loss with class probabilities 
        loss = loss_fn(upscale_logits(logits),
                    yb.squeeze(1).to(torch.long)
        )

        # accumulate the loss of the epoch
        cur_loss += loss.item()

        # Backward prop
        loss.backward()

        # Update the parameters
        optimizer.step()
        steps += 1

        if steps%eval_steps == 0:
            # Compute mean intersection over union in the valset
            val_preds, val_labels, val_logits = get_pred_label(model, 
                                                               val_loader, 
                                                               device, 
                                                               return_logits=True)
            val_set_iou = compute_iou(val_preds, val_labels)
            miou_mean =  torch.mean(val_set_iou).item()
            miou.append(val_set_iou.detach().cpu().tolist())
            val_lossi.append(loss_fn(upscale_logits(val_logits), 
                                     val_labels.squeeze(1).to(torch.long)).item())
            
            # save check point
            if miou_mean > best_miou:
                best_miou = miou_mean
                print(f"Saving a checkpoint that reach a mIoU of {best_miou:.4f}")
                ckpt = {
                'epochs': epochs,
                'cur_steps': steps,
                'best_miou': best_miou,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }
                torch.save(ckpt, './ckpt/single-segmentation.pth')
                print("Checkpoint saved!")

            # Print current mIoU
            print(f" -- Validation loss at step {val_lossi[-1]:.4f} | mIoU at step #{steps}: {miou_mean:.4f}")

    # save the average loss per epoch
    lossi[idx] = cur_loss / len(train_loader)

    print(f" -- Training loss at epoch {idx}: {lossi[idx]:.4f}")

# compute running time
end_time = time.time()
print(f"--- {end_time - start_time} seconds ---")


df_miou = pd.DataFrame(miou)

df_miou.to_csv('./results/miou_'+ str(rep)+'.csv')
