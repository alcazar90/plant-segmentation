import wandb

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import upscale_logits, flatten_logits


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