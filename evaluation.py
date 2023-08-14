import wandb
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from utils import upscale_logits, flatten_logits

def get_pred_label(model, dl, device, alternative_masks=False):
    """
    Collect the images, labels, and alternative masks from a dataloader and
    return them as torch.tensor objects with shape (B, H, W). In addition, 
    compute the predicted masks and probabilities from the model and return them
    as torch.tensor.

    Args:
        model: a torch.nn.Module object.
        dl: a torch.utils.data.DataLoader object.
        device: a torch.device object.
        alternative_masks: a boolean indicating if the dataloader contains
            alternative masks (see PlantDataset option).

    Returns:
        images: a torch.tensor with shape (B, C, H, W) containing the images
            in the dataloader
        labels: a torch.tensor with shape (B, H, W) containing the ground truth
            masks in the dataloader
        preds: a torch.tensor with shape (B, H, W) containing the predicted 
            masks in the dataloader
        probs: a torch.tensor with shape (B, H, W) containing the probabilities 
            of the predicted masks in the dataloader
        alternative: a torch.tensor with shape (B, H, W) containing the 
            alternative masks in the dataloader

    Examples:

        >>> val_img, val_labels, val_preds, val_probs = get_pred_label(model, 
                                                                       val_loader, 
                                                                       device)
        
                                                                    

        >>> _, val_labels, val_preds, _, val_alternative = get_pred_label(model,
                                                                          val_loader,
                                                                          device,
                                                                          alternative_masks=True)
    """
    model.eval() 
    images = []
    preds = []
    labels = []
    probs = []
    alternative = [] if alternative_masks else None

    for batch in tqdm(dl):
        xb = batch[0].to(device)
        yb = batch[1].to(device)
        if alternative_masks:
            zb = batch[2].to(device)

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

        # Check if alternative masks are provided
        if alternative_masks:
            alternative.append(zb.detach().cpu())
    
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    images = torch.cat(images, dim=0)
    probs = torch.cat(probs, dim=0)

    if (len(labels.shape) > 3):
        labels.squeeze_(1)

    if alternative_masks:
        alternative = torch.cat(alternative, dim=0)
        if (len(alternative.shape) > 3):
            alternative.squeeze_(1)
        return images, labels, preds, probs, alternative

    return images, labels, preds, probs

def compute_iou(preds, labels, eps=1e-6):
    """
    Compute the intersection over union metric for each observation and
    return (B,) tensor with the IoU for each observation.

    Args:
        preds: (B, H, W) tensor with the predicted masks.
        labels: (B, H, W) tensor with the ground truth masks.

    Returns:
        torch.tensor with shape (B,) containing the IoU for each observation.
    """
    inter = torch.logical_and(preds, labels)
    union = torch.logical_or(preds, labels)
    iou = torch.div(torch.sum(inter, dim=(1, 2)),  
                    torch.sum(union, dim=(1, 2)) + eps)
    return iou

def compute_intersection_over_union(logits, labels, eps=1e-6):
    """ 
    Compute the intersection over union metric between the logits and the labels.
    It obtain the predicted masks from the logits by picking the label with the
    highest probability.

    B: batch size
    C: number of classes
    H: height of the image
    W: width of the image

    Args:
        logits: torch.tensor with shape (B, C, H, W) containing the logits 
            return by the model
        labels: torch.tensor with shape (B, H, W) containing the labels
            return by the DataLoader format

    Returns: 
        float with the ratio of intersection over union
    """
    # remove gradient graph and pass to cpu
    preds = upscale_logits(logits.detach().cpu())
    labels = labels.detach().cpu()

    # normalize logits and get probabilities
    preds = torch.argmax(torch.softmax(preds, dim=1), dim=1)

    # collapse the labels channels into a single one, from multiple one-hot 
    # channels
    labels = torch.argmax(labels, dim=1)

    # compute intersection and union between prediction and labels
    intersection = torch.logical_and(preds, labels)
    union = torch.logical_or(preds, labels)

    iou = (torch.div(torch.sum(intersection, dim=(1,2)) + eps, 
                     (torch.sum(union, dim=(1,2)) + eps)).sum() / logits.shape[0]).item()

    return iou
