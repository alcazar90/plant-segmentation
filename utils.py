import re
import os
import random
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt 
import torch.nn as nn
from PIL import Image

def display_masks(images):
    """ Stack PIL images horizontally """
    images = [Image.open(p) for p in images]
    images = [img.resize((225, 225)) for img in images]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    return new_im

def extract_ids_from_name(string) -> tuple:
    """ Helper function to retrieve ids from the image name """

    # expected to be one '_' in the name structure: cwt4_55.jpg
    two_part_string = string.split('_')

    # left digits are the 'id' original img'
    id_original_img = int(re.findall(r'\d+$', two_part_string[0])[0])

    # right digits are the 'id after split'
    #id_after_split = int(two_part_string[1].split('.')[0])
    id_after_split = int(re.findall(r'\d+', two_part_string[1])[0])

    return id_original_img, id_after_split

def extract_tag_from_name(string) -> str:
    """ Helper function to retrieve the class tag from the image name 


        Expected name structure from label studio:
            data/cwt/original_labeled/22/task-4-annotation-17-bhy-1-tag-[NAME OF THE TAG]-0.jpg

        Goal: get the name of the tag
    """
    return string.split('/')[-1].split('-')[-2]

def upscale_logits(logit_outputs, res=512):
  """Escala los logits a (4W)x(4H) para recobrar dimensiones originales del input"""
  return nn.functional.interpolate(
      logit_outputs,
      size=(res,res),
      mode='bilinear',
      align_corners=False
  )

def flatten_logits(logits):
  return logits.contiguous().view(logits.shape[0], -1)

def set_seed(seed: int = 42313988) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def display_batch_masks(bmasks, caption=None):
    """
        Display a batch of images with their predicted masks in a grid.

        bmasks.shape -> (B, H, W)
    """
    grid = torchvision.utils.make_grid(
        bmasks,
        nrow=bmasks.shape[0],
        padding=8,
        pad_value=90
    )
    fig = plt.figure(figsize=(16, 12))
    plt.imshow(np.uint8(grid.permute(1,2,0).numpy()) * 255, cmap='Greys');
    plt.axis('off');
    if caption:
        plt.title(caption);