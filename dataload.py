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


class MassachusettsBuildingsDataset(Dataset):
    """
    Dataset class for Massachusetts Buildings Dataset with improved error handling
    """
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        image_size: Tuple[int, int] = (512, 512),
        transform: Optional[A.Compose] = None,
        preprocessing: Optional[A.Compose] = None,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.transform = transform
        self.preprocessing = preprocessing

        # Validate paths and remove invalid pairs
        self.image_paths, self.mask_paths = self._validate_paths()
        
        if len(self.image_paths) == 0:
            raise RuntimeError("No valid image-mask pairs found")

    def _validate_paths(self) -> Tuple[List[str], List[str]]:
        """Validate and filter paths, returning only valid pairs"""
        valid_img_paths = []
        valid_mask_paths = []
        
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            if os.path.exists(img_path) and os.path.exists(mask_path):
                try:
                    # Try to open both files to ensure they're valid
                    with Image.open(img_path) as img:
                        img_size = img.size
                    with Image.open(mask_path) as mask:
                        mask_size = mask.size
                    
                    # Check if sizes match
                    if img_size == mask_size:
                        valid_img_paths.append(img_path)
                        valid_mask_paths.append(mask_path)
                except Exception as e:
                    print(f"Error loading {img_path} or {mask_path}: {str(e)}")
                    continue
            
        return valid_img_paths, valid_mask_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            # Load image
            image = self._load_image(self.image_paths[idx])
            mask = self._load_mask(self.mask_paths[idx])
            
            # Ensure images are the right size
            image, mask = self._resize_if_needed(image, mask)
            
            # Apply transformations
            if self.transform is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']

            # Apply preprocessing
            if self.preprocessing is not None:
                preprocessed = self.preprocessing(image=image, mask=mask)
                image = preprocessed['image']
                mask = preprocessed['mask']

            # Final checks
            if torch.is_tensor(image):
                image = image.float()
            if torch.is_tensor(mask):
                mask = mask.float()

            return image, mask

        except Exception as e:
            print(f"Error processing index {idx}: {str(e)}")
            # Return a zero tensor of the correct shape as a fallback
            return (torch.zeros((3, *self.image_size)), 
                   torch.zeros(self.image_size))

    def _load_image(self, path: str) -> np.ndarray:
        """Safely load an image"""
        try:
            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"Failed to load image: {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            raise ValueError(f"Error loading image {path}: {str(e)}")

    def _load_mask(self, path: str) -> np.ndarray:
        """Safely load a mask"""
        try:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask: {path}")
            mask = (mask > 0).astype(np.float32)
            return mask
        except Exception as e:
            raise ValueError(f"Error loading mask {path}: {str(e)}")

    def _resize_if_needed(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resize image and mask if they don't match the target size"""
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size[::-1], interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.image_size[::-1], interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0.5).astype(np.float32)
        return image, mask









def get_preprocessing() -> A.Compose:
    """Get preprocessing transforms"""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_transforms(image_size: Tuple[int, int], is_training: bool = True) -> A.Compose:
    """Get transforms for training or validation"""
    if is_training:
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=45,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT
            )
        ])
    else:
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1])
        ])

def create_dataloaders(
    img_train_paths: List[str],
    gts_train_paths: List[str],
    img_val_paths: List[str],
    gts_val_paths: List[str],
    image_size: Tuple[int, int] = (512, 512),
    batch_size: int = 4,
    num_workers: int = 0  # Set to 0 for debugging
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders with error handling"""
    
    try:
        # Create datasets
        train_dataset = MassachusettsBuildingsDataset(
            image_paths=img_train_paths,
            mask_paths=gts_train_paths,
            image_size=image_size,
            transform=get_transforms(image_size, is_training=True),
            preprocessing=get_preprocessing()
        )

        val_dataset = MassachusettsBuildingsDataset(
            image_paths=img_val_paths,
            mask_paths=gts_val_paths,
            image_size=image_size,
            transform=get_transforms(image_size, is_training=False),
            preprocessing=get_preprocessing()
        )

        # Create dataloaders with error handling
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if num_workers > 0 else False,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if num_workers > 0 else False,
        )

        return train_loader, val_loader

    except Exception as e:
        print(f"Error creating dataloaders: {str(e)}")
        raise





if __name__ == "__main__":
    # Test the dataloaders with error handling
    try:
        # Create dataloaders with num_workers=0 for debugging
        train_loader, val_loader = create_dataloaders(
            img_train_paths=img_train_paths,
            gts_train_paths=gts_train_paths,
            img_val_paths=img_val_paths,
            gts_val_paths=gts_val_paths,
            image_size=(512, 512),
            batch_size=4,
            num_workers=0  # Set to 0 for debugging
        )

        # Test loading a batch
        for images, masks in train_loader:
            print(f"Successfully loaded batch:")
            print(f"Image batch shape: {images.shape}")
            print(f"Mask batch shape: {masks.shape}")
            break

    except Exception as e:
        print(f"Error in main execution: {str(e)}")