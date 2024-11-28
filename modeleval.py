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
from train import train_model

class ModelEvaluator:
    def __init__(
        self,
        model: nn.Module,
        device: str,
        criterion: nn.Module,
        project_name: str = "building-segmentation",
    ):
        self.model = model
        self.device = device
        self.criterion = criterion

        # Initialize wandb
        wandb.init(project=project_name, config={
            "architecture": model.__class__.__name__,
            "device": device,
            "criterion": criterion.__class__.__name__
        })

        # Initialize metrics storage
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_metrics: List[Dict[str, float]] = []
        self.val_metrics: List[Dict[str, float]] = []

    def calculate_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Calculate various segmentation metrics"""
        # Convert predictions to binary
        pred_binary = (outputs > threshold).float().cpu().numpy()
        targets_binary = targets.cpu().numpy()

        # Flatten the predictions and targets
        pred_flat = pred_binary.reshape(-1)
        targets_flat = targets_binary.reshape(-1)

        # Calculate metrics
        metrics = {
            'precision': precision_score(targets_flat, pred_flat, zero_division=1),
            'recall': recall_score(targets_flat, pred_flat, zero_division=1),
            'f1': f1_score(targets_flat, pred_flat, zero_division=1),
            'iou': jaccard_score(targets_flat, pred_flat, zero_division=1)
        }

        return metrics

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch and return average loss and metrics"""
        self.model.train()
        total_loss = 0
        batch_metrics = []

        progress_bar = tqdm(train_loader, desc="Training")
        for images, masks in progress_bar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            metrics = self.calculate_metrics(outputs, masks)
            batch_metrics.append(metrics)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'iou': f"{metrics['iou']:.4f}"
            })

        # Calculate average metrics
        avg_metrics = {
            key: np.mean([m[key] for m in batch_metrics])
            for key in batch_metrics[0].keys()
        }
        avg_loss = total_loss / len(train_loader)

        return avg_loss, avg_metrics

    def validate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        """Validate the model and return average loss and metrics"""
        self.model.eval()
        total_loss = 0
        batch_metrics = []

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            for images, masks in progress_bar:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                total_loss += loss.item()
                metrics = self.calculate_metrics(outputs, masks)
                batch_metrics.append(metrics)

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'iou': f"{metrics['iou']:.4f}"
                })

        # Calculate average metrics
        avg_metrics = {
            key: np.mean([m[key] for m in batch_metrics])
            for key in batch_metrics[0].keys()
        }
        avg_loss = total_loss / len(val_loader)

        return avg_loss, avg_metrics

    def train_and_evaluate(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        num_epochs: int,
        save_path: str = 'best_model.pth'
    ) -> None:
        """Train and evaluate the model for specified number of epochs"""
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, optimizer)
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metrics)

            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)

            # Update learning rate
            scheduler.step(val_loss)

            # Log metrics to wandb
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_iou': train_metrics['iou'],
                'val_iou': val_metrics['iou'],
                'train_f1': train_metrics['f1'],
                'val_f1': val_metrics['f1'],
                'learning_rate': optimizer.param_groups[0]['lr']
            })

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'train_metrics': self.train_metrics,
                    'val_metrics': self.val_metrics
                }, save_path)
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")

            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_metrics['iou']:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_metrics['iou']:.4f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

    def plot_metrics(self) -> None:
        """Plot training and validation metrics"""
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot loss curves
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot IoU curves
        train_iou = [m['iou'] for m in self.train_metrics]
        val_iou = [m['iou'] for m in self.val_metrics]
        ax2.plot(epochs, train_iou, 'b-', label='Training IoU')
        ax2.plot(epochs, val_iou, 'r-', label='Validation IoU')
        ax2.set_title('IoU Curves')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('IoU')
        ax2.legend()

        # Plot F1 curves
        train_f1 = [m['f1'] for m in self.train_metrics]
        val_f1 = [m['f1'] for m in self.val_metrics]
        ax3.plot(epochs, train_f1, 'b-', label='Training F1')
        ax3.plot(epochs, val_f1, 'r-', label='Validation F1')
        ax3.set_title('F1 Score Curves')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('F1 Score')
        ax3.legend()

        # Plot Precision-Recall curves
        train_precision = [m['precision'] for m in self.train_metrics]
        train_recall = [m['recall'] for m in self.train_metrics]
        val_precision = [m['precision'] for m in self.val_metrics]
        val_recall = [m['recall'] for m in self.val_metrics]

        ax4.plot(train_recall, train_precision, 'b-', label='Training')
        ax4.plot(val_recall, val_precision, 'r-', label='Validation')
        ax4.set_title('Precision-Recall Curves')
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.legend()

        plt.tight_layout()
        wandb.log({"metrics_plot": wandb.Image(plt)})
        plt.close()

# Example usage:
if __name__ == "__main__":
    # Initialize model, criterion, optimizer, and scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaUNet(in_channels=3, out_channels=1, features=(32, 64, 128)).to(device)
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )

    # Initialize evaluator
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        criterion=criterion,
        project_name="building-segmentation"
    )

    # Train and evaluate
    evaluator.train_and_evaluate(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=100,
        save_path='best_mamba_unet.pth'
    )

    # Plot final metrics
    evaluator.plot_metrics()

    # Close wandb run
    wandb.finish()