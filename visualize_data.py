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




image = Image.open(img_train_paths[0])
label = Image.open(gts_train_paths[0])

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Input Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Ground Truth Label")
plt.imshow(label, cmap='gray')
plt.show()


def visualize_transformed_dataset(
    dataset: MassachusettsBuildingsDataset,
    num_samples: int = 5
) -> None:
    """
    Visualize raw and transformed samples from the dataset.

    Args:
        dataset (MassachusettsBuildingsDataset): Dataset to visualize.
        num_samples (int): Number of samples to visualize.
    """
    # Select random indices to visualize
    indices = random.sample(range(len(dataset)), num_samples)
    
    for idx in indices:
        try:
            # Get raw image and mask
            raw_image = cv2.imread(dataset.image_paths[idx])
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            raw_mask = cv2.imread(dataset.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            
            # Get transformed image and mask
            transformed_image, transformed_mask = dataset[idx]
            
            # Denormalize the transformed image for visualization
            transformed_image = TF.normalize(
                transformed_image,
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ).permute(1, 2, 0).numpy()
            
            # Convert tensors to numpy arrays
            transformed_image = (transformed_image * 255).astype(np.uint8)
            transformed_mask = transformed_mask.numpy()
            
            # Plot original and transformed images
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            
            axs[0, 0].imshow(raw_image)
            axs[0, 0].set_title("Raw Image")
            axs[0, 0].axis("off")
            
            axs[0, 1].imshow(raw_mask, cmap="gray")
            axs[0, 1].set_title("Raw Mask")
            axs[0, 1].axis("off")
            
            axs[1, 0].imshow(transformed_image)
            axs[1, 0].set_title("Transformed Image")
            axs[1, 0].axis("off")
            
            axs[1, 1].imshow(transformed_mask, cmap="gray")
            axs[1, 1].set_title("Transformed Mask")
            axs[1, 1].axis("off")
            
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            print(f"Error visualizing sample {idx}: {str(e)}")






if __name__ == "__main__":
    # Load the training dataset for visualization
    train_dataset = MassachusettsBuildingsDataset(
        image_paths=img_train_paths,
        mask_paths=gts_train_paths,
        image_size=(512, 512),
        transform=get_transforms(image_size=(512, 512), is_training=True),
        preprocessing=get_preprocessing(),
    )
    
    print("Visualizing dataset...")
    visualize_transformed_dataset(train_dataset, num_samples=5)
