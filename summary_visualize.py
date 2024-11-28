import os 
import numpy as np 
import pandas as pd 
import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import  transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF 
from PIL import Image
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from typing import List,Tuple,Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import warnings
warnings.filterwarnings('ignore')
import wandb
import cv2
from dataload import MassachusettsBuildingsDataset,train_loader,val_loader


root = "./data/png"

img_train_dir = os.path.join(root, "train")
gts_train_dir = os.path.join(root, "train_labels")

img_val_dir = os.path.join(root, "val")
gts_val_dir = os.path.join(root, "val_labels")

img_test_dir = os.path.join(root, "test")
gts_test_dir = os.path.join(root, "test_labels")


img_train_paths = [os.path.join(img_train_dir, image_id) for image_id in sorted(os.listdir(img_train_dir))]
gts_train_paths = [os.path.join(gts_train_dir, image_id) for image_id in sorted(os.listdir(gts_train_dir))]

img_val_paths = [os.path.join(img_val_dir, image_id) for image_id in sorted(os.listdir(img_val_dir))]
gts_val_paths = [os.path.join(gts_val_dir, image_id) for image_id in sorted(os.listdir(gts_val_dir))]

img_test_paths = [os.path.join(img_test_dir, image_id) for image_id in sorted(os.listdir(img_test_dir))]
gts_test_paths = [os.path.join(gts_test_dir, image_id) for image_id in sorted(os.listdir(gts_test_dir))]


print(len(img_train_paths), len(gts_train_paths))
print(img_train_paths[5], gts_train_paths[5])

print(len(img_val_paths), len(gts_val_paths))
print(img_val_paths[2], gts_val_paths[2])

print(len(img_test_paths), len(gts_test_paths))
print(img_test_paths[1], gts_test_paths[1])



def dataset_summary(image_paths: List[str], mask_paths: List[str]) -> None:
    """
    Summarize key statistics about the dataset.
    
    Args:
        image_paths (List[str]): Paths to the dataset images.
        mask_paths (List[str]): Paths to the dataset masks.
    """
    print("Exploring Dataset Summary...")
    num_samples = len(image_paths)
    print(f"Total Samples: {num_samples}")
    
    # Variables to collect statistics
    image_shapes = []
    mask_shapes = []
    building_pixel_counts = []
    total_pixel_counts = []

    for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=num_samples):
        # Load image and mask
        try:
            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Image and mask shapes
            image_shapes.append(image.shape[:2])
            mask_shapes.append(mask.shape)

            # Building pixel statistics
            building_pixels = np.sum(mask > 0)
            total_pixels = mask.size
            building_pixel_counts.append(building_pixels)
            total_pixel_counts.append(total_pixels)

        except Exception as e:
            print(f"Error processing {img_path} or {mask_path}: {str(e)}")

    # Calculate statistics
    print("\nImage Shape Statistics:")
    print(f"- Min Dimensions: {np.min(image_shapes, axis=0)}")
    print(f"- Max Dimensions: {np.max(image_shapes, axis=0)}")
    print(f"- Mean Dimensions: {np.mean(image_shapes, axis=0)}")

    print("\nMask Shape Statistics:")
    print(f"- Min Dimensions: {np.min(mask_shapes, axis=0)}")
    print(f"- Max Dimensions: {np.max(mask_shapes, axis=0)}")
    print(f"- Mean Dimensions: {np.mean(mask_shapes, axis=0)}")

    print("\nBuilding Pixel Statistics:")
    print(f"- Total Building Pixels: {np.sum(building_pixel_counts)}")
    print(f"- Mean Building Pixels per Mask: {np.mean(building_pixel_counts)}")
    print(f"- Median Building Pixels per Mask: {np.median(building_pixel_counts)}")
    print(f"- Building Pixel Coverage (%): {100 * np.sum(building_pixel_counts) / np.sum(total_pixel_counts):.2f}")



def visualize_mask_overlay(image_paths: List[str], mask_paths: List[str], num_samples: int = 5) -> None:
    """
    Visualize random samples with masks overlaid on images.
    
    Args:
        image_paths (List[str]): Paths to the dataset images.
        mask_paths (List[str]): Paths to the dataset masks.
        num_samples (int): Number of random samples to visualize.
    """
    indices = random.sample(range(len(image_paths)), num_samples)

    for idx in indices:
        try:
            # Load image and mask
            image = cv2.imread(image_paths[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            
            # Create overlay
            overlay = image.copy()
            overlay[mask > 0] = [255, 0, 0]  # Red overlay for building pixels
            
            # Plot
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 3, 1)
            plt.title("Original Image")
            plt.imshow(image)
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title("Mask")
            plt.imshow(mask, cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title("Overlay")
            plt.imshow(overlay)
            plt.axis("off")
            
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error visualizing sample {idx}: {str(e)}")




print("Summary for Training Set:")
dataset_summary(img_train_paths, gts_train_paths)

print("\nVisualizing Overlayed Masks for Training Set:")
visualize_mask_overlay(img_train_paths, gts_train_paths, num_samples=5)