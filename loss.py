import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from typing import Optional, Tuple, List, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import torchvision.transforms.functional as TF
from dataload import MassachusettsBuildingsDataset, train_loader, val_loader,img_test_dir,images,img_train_dir,img_val_dir
!wandb login 
from model import model,MambaDownBlock,MambaUNet,train_model,MambaUpBlock,SelectiveSSM


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation"""
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.squeeze(1)  # Remove channel dimension for binary segmentation
        intersection = (pred * target).sum(dim=(1, 2))
        union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()