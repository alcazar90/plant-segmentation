import wandb
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from utils import upscale_logits, flatten_logits

def get_pred_label(model, dl, device):
    """
        Given a model and a dataloader, return (B, H, W) tensor with the predicted
        masks and (B, H, W) tensor with the ground truth masks for each image in
        the dataset.
    """
    model.eval() 
    images = []
    preds = []
    labels = []
    probs = [] 

    for xb, yb in tqdm(dl):
        xb = xb.to(device)
        yb = yb.to(device)

        # Perform a forward pass
        logits = model(xb)["logits"]
        
        # Upscale logits to original size
        logits = upscale_logits(logits.detach().cpu())

        # Normalize logits, get probabilities, and pick the label with the highest
        # probability
        prob_masks = torch.softmax(logits, dim=1)
        masks = torch.argmax(prob_masks, dim=1)
        probs.append(prob_masks[:, 1, :, :].detach().cpu())

        # Append batch predicted masks
        preds.append(masks.detach().cpu())

        # Save images and labels/ground truth
        images.append(xb.detach().cpu())
        labels.append(yb.detach().cpu())
    
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    images = torch.cat(images, dim=0)
    probs = torch.cat(probs, dim=0)

    if (len(labels.shape) > 3):
        labels.squeeze_(1)

    return images, labels, preds, probs


def compute_iou(preds, labels, eps=1e-6):
    """

    Compute the intersection over union metric for each observation and
    return (B,) tensor with the IoU for each observation.

      Args:
        preds: (B, H, W) tensor with the predicted masks
        labels: (B, H, W) tensor with the ground truth masks

    """
    inter = torch.logical_and(preds, labels)
    union = torch.logical_or(preds, labels)
    iou = torch.div(torch.sum(inter, dim=(1, 2)),  
                    torch.sum(union, dim=(1, 2)) + eps)
    return iou

def compute_intersection_over_union(logits, targets, eps=1e-6):
    """ 
    Compute the intersection over union between the logits and the targets

    B: batch size
    C: number of classes
    H: height of the image
    W: width of the image

    Args:
        logits: torch.tensor with shape (B, C, H, W) containing the logits 
            return by the model
        targets: torch.tensor with shape (B, C, H, W) containing the targets
            return by the DataLoader format

    Returns: a float with the ratio of intersection over union
    """
    # eliminar grafo de gradiente y pasar a cpu
    preds = upscale_logits(logits.detach().cpu())
    targets = targets.detach().cpu()

    # normalizar logits y obtener probabilidades
    preds = torch.argmax(torch.softmax(preds, dim=1), dim=1)

    # colapsar los canales del target en uno solo, de multiples one-hot channels
    # a un solo canal con valores enteros
    targets = torch.argmax(targets, dim=1)

    # computar intersección y unión entre conjuntos de prediccion y targets
    intersection = torch.logical_and(preds, targets)
    union = torch.logical_or(preds, targets)

    # computar ratio de intersección sobre la unión 
    iou = (torch.div(torch.sum(intersection, dim=(1,2)) + eps, 
                     (torch.sum(union, dim=(1,2)) + eps)).sum() / logits.shape[0]).item()

    return iou
