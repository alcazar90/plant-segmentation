import wandb

import matplotlib.pyplot as plt

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import upscale_logits, flatten_logits

def get_pred_label(model, dl, device):
    """
        Given a model and a dataloader, return (B, H, W) tensor with the predicted
        masks and (B, H, W) tensor with the ground truth masks for each image in
        the dataset.
    """
    model.eval() 
    preds = []
    labels = []

    for xb, yb in tqdm(dl):
        xb = xb.to(device)
        yb = yb.to(device)

        # Perform a forward pass
        logits = model(xb)["logits"]

        # Upscale logits to original size
        masks = upscale_logits(logits.detach().cpu())

        # Normalize logits, get probabilities, and pick the label with the highest
        # probability
        masks = torch.argmax(torch.softmax(masks, dim=1), dim=1)

        # Append batch predicted masks
        preds.append(masks)

        # Save labels/ground truth
        labels.append(yb.detach().cpu())
    
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)

    if (len(labels.shape) > 3):
        labels.squeeze_(1)

    return preds, labels


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


def validate_model(model, valid_dl, loss_fn, log_images=False, num_classes=2):
  """Compute performance of the model on the validation dataset and log a wandb.Table"""
  #cmap = plt.get_cmap('viridis')
  model.eval()
  val_loss = 0.
  iou = 0.
  with torch.inference_mode():
    correct = 0

    iou_by_example = torch.zeros(len(valid_dl.dataset)) # tensor with length equals to the number of validation examples
    for i, (images, masks) in enumerate(valid_dl):
      images, masks = images.to(device), masks.to(device)
      masks=flatten_logits(torch.where(masks > 0.0, 1.0, 0.0))

      # Forward pass ➡
      logits = model(images)["logits"]
      probs = torch.softmax(upscale_logits(logits), dim=1)
      _, predicted = torch.max(probs.data, dim=1)
      probs = probs[:, 1, :, :]
      preds = flatten_logits(probs)
      val_loss+= loss_fn(preds, masks).item()*masks.size(0)

      # Compute pixel accuracy and accumulate
      correct += (flatten_logits(predicted) == masks).sum().item()

      # Compute IoU and accumulate
      mask2d=masks.view(masks.shape[0], predicted.shape[1], -1)
      intersection = torch.logical_and(mask2d, predicted)
      union = torch.logical_or(mask2d, predicted)
      iou += (torch.div(torch.sum(intersection, dim=(1,2)) + 1e-6, (torch.sum(union, dim=(1,2)) + 1e-6)).sum()/predicted.shape[0]).item()

      # tensor with IoU for every example (batch_size x 1)
      iou_by_example = intersection.sum(dim=(1,2), keepdim=False) / (union.sum(dim=(1,2), keepdim=False) + 1e-6)

      
      # Log validation predictions and images to the dashboard
      if log_images:
        if i == 0:
          # 🐝 Create a wandb Table to log images, labels and predictions to
          table = wandb.Table(columns=["image", "mask", "pred_mask", "probs", "iou"])
          for img, mask, pred, prob, iou_metric in zip(images.to("cpu"), masks.to("cpu"), predicted.to("cpu"), probs.to("cpu"), iou_by_example.to("cpu")):
            plt.imshow(prob.detach().cpu());
            #plt.imshow(cmap(prob].detach().cpu().numpy())[:,:,:3])
            plt.axis("off");
            plt.tight_layout();
            table.add_data(wandb.Image(img.permute(1,2,0).numpy()), 
                           wandb.Image(mask.view(img.shape[1:]).unsqueeze(2).numpy()),
                           wandb.Image(np.uint8(pred.unsqueeze(2).numpy())*255),
                           #wandb.Image(Image.fromarray((cmap(prob.detach().cpu().numpy())[:,:,:]*255).astype(np.uint8)[:,:,3]))
                           wandb.Image(plt),
                           iou_metric
                           )
    if log_images:
      wandb.log({"val_table/predictions_table":table}, commit=False)

  return (
        val_loss / len(valid_dl.dataset), 
        correct / (len(valid_dl.dataset)*512**2),
        iou / (i+1)
  )