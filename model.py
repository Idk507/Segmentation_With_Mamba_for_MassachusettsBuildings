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



class SelectiveSSM(nn.Module):
    """Simplified Selective State Space Model block"""
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.channels = channels

        # Simplified SSM without sequential processing
        self.conv1 = nn.Conv2d(channels, channels * 2, 1)
        self.dw_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.norm = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, C, H, W]
        projected = self.conv1(x)
        content, gate = projected.chunk(2, dim=1)

        # Simplified sequential processing using depthwise conv
        out = self.dw_conv(content)
        out = self.norm(out)
        out = F.gelu(out)
        out = self.conv2(out)

        # Apply gating
        out = out * torch.sigmoid(gate)
        return self.dropout(out)

class MambaDownBlock(nn.Module):
    """Downsampling block with simplified Mamba layer"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.mamba = SelectiveSSM(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.mamba(x)
        x = F.relu(self.bn2(self.conv2(x)))
        return self.pool(x)

class MambaUpBlock(nn.Module):
    """Upsampling block with simplified Mamba layer"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.mamba = SelectiveSSM(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.mamba(x)
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class MambaUNet(nn.Module):
    """U-Net architecture with simplified Mamba blocks"""
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: tuple = (32, 64, 128)
    ):
        super().__init__()

        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )

        # Encoder
        self.down_blocks = nn.ModuleList([
            MambaDownBlock(features[i], features[i + 1])
            for i in range(len(features) - 1)
        ])

        # Decoder
        self.up_blocks = nn.ModuleList([
            MambaUpBlock(features[i + 1], features[i])
            for i in range(len(features) - 1)
        ][::-1])

        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial convolution
        x = self.init_conv(x)

        # Encoder path with skip connections
        skips = []
        for block in self.down_blocks:
            skips.append(x)
            x = block(x)

        # Decoder path
        for block, skip in zip(self.up_blocks, skips[::-1]):
            x = block(x, skip)

        # Final convolution
        return torch.sigmoid(self.final_conv(x))

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

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> None:
    print(f"Using device: {device}")
    model = model.to(device)
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=True
        )

        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, 'best_mamba_unet.pth')
            print("Saved best model checkpoint")

        print("-" * 50)


# Initialize model
model = MambaUNet(
    in_channels=3,
    out_channels=1,
    features=(32, 64, 128)
)

#!wandb login



