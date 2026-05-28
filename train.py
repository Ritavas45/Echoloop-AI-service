import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Try importing sklearn metrics, fallback gracefully if not installed
try:
    from sklearn.metrics import f1_score, confusion_matrix, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from dataset import PhoneConditionDataset, get_transforms
from model import LateFusionEfficientNet

class EarlyStopping:
    """
    Early stopping helper to stop training if validation score (Macro F1) 
    stops improving after a given patience.
    """
    def __init__(self, patience=7, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.delta:
            self.counter += 1
            print(f"[EarlyStopping] Score did not improve. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print(f"[EarlyStopping] Score improved from {self.best_score:.4f} to {val_score:.4f}!")
            self.best_score = val_score
            self.counter = 0

def print_confusion_matrix(cm, classes):
    """
    Prints a text-based confusion matrix beautifully formatted for command line viewing.
    """
    header = f"{'Actual \\ Pred':<15} | " + " | ".join([f"{cls:<10}" for cls in classes])
    border = "-" * len(header)
    print("\n" + border)
    print(header)
    print(border)
    for i, row in enumerate(cm):
        row_str = f"{classes[i]:<15} | " + " | ".join([f"{val:<10}" for val in row])
        print(row_str)
    print(border + "\n")

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass: model outputs average class probabilities
        # Shape: [BatchSize, NumClasses]
        avg_probs = model(images)
        
        # Calculate loss using NLLLoss on log probabilities
        # log of probabilities is the correct loss input when model outputs softmax probabilities
        log_probs = torch.log(avg_probs + 1e-15)
        loss = F.nll_loss(log_probs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        # Get predictions
        preds = avg_probs.argmax(dim=-1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate(model, dataloader, device, classes):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        
        avg_probs = model(images)
        log_probs = torch.log(avg_probs + 1e-15)
        loss = F.nll_loss(log_probs, targets)
        
        running_loss += loss.item() * images.size(0)
        
        preds = avg_probs.argmax(dim=-1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
    val_loss = running_loss / total
    val_acc = correct / total
    
    # Calculate advanced metrics
    macro_f1 = 0.0
    per_class_f1 = []
    
    if SKLEARN_AVAILABLE:
        macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
        cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(classes))))
        
        # Print metrics summary
        print(f"\nValidation Classification Report:")
        print(classification_report(all_targets, all_preds, target_names=classes, zero_division=0))
        print("Confusion Matrix:")
        print_confusion_matrix(cm, classes)
    else:
        print("\nWarning: scikit-learn is not installed. Skipping advanced F1-score & Confusion Matrix printing.")
        print("Install scikit-learn to get per-class statistics and confusion matrix.")
        # Fallback simplistic calculation
        macro_f1 = val_acc
        per_class_f1 = [0.0] * len(classes)
        
    return val_loss, val_acc, macro_f1, per_class_f1

def main():
    parser = argparse.ArgumentParser(description="Train Late Fusion EfficientNet-B3 Classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root folder containing train/val directories")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--img_size", type=int, default=300, help="EfficientNet image resolution size")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader CPU workers")
    args = parser.parse_args()

    # Create checkpoint folder
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. Device Setup (Detect Apple MPS, CUDA, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon GPU (MPS)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    print(f"Training device: {device} ({device_name})")

    # 2. Get Transforms and Datasets
    train_transform, val_transform = get_transforms(args.img_size)
    
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Missing 'train' or 'val' subdirectory under data directory: {args.data_dir}")

    train_dataset = PhoneConditionDataset(train_dir, transform=train_transform)
    val_dataset = PhoneConditionDataset(val_dir, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 3. Model Setup
    classes = ['Reuse', 'Refurbish', 'Repair', 'Recycle']
    print("Initializing Late Fusion EfficientNet-B3 model...")
    model = LateFusionEfficientNet(num_classes=len(classes), pretrained=True)
    model = model.to(device)

    # 4. Optimization Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    
    # ReduceLROnPlateau: monitors macro F1 (we want to maximize it, so mode='max')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=2, 
        threshold=1e-4
    )
    
    early_stopping = EarlyStopping(patience=args.patience)

    # 5. Training Loop
    best_macro_f1 = -1.0
    print("Starting training loop...")
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Get current learning rate before train step
        current_lr = optimizer.param_groups[0]['lr']
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_macro_f1, per_class_f1 = validate(model, val_loader, device, classes)
        
        # Learning Rate step based on validation macro F1
        scheduler.step(val_macro_f1)
        new_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - start_time
        
        # Print epoch summary
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}% - Val Macro F1: {val_macro_f1:.4f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        
        if new_lr != current_lr:
            print(f"--> Learning rate reduced from {current_lr:.2e} to {new_lr:.2e}")
            
        for idx, cls_name in enumerate(classes):
            print(f"  - {cls_name} F1: {per_class_f1[idx]:.4f}")
            
        # 6. Checkpointing based on macro F1
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_macro_f1': max(best_macro_f1, val_macro_f1),
            'val_macro_f1': val_macro_f1,
            'classes': classes
        }
        
        # Save latest model anyway
        latest_path = os.path.join(args.save_dir, 'latest_model.pth')
        torch.save(checkpoint, latest_path)
        
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"==> Saved new best model checkpoint to {best_path} (Macro F1: {best_macro_f1:.4f})")
            
        # Early Stopping check
        early_stopping(val_macro_f1)
        if early_stopping.early_stop:
            print("Early stopping triggered. Training stopped.")
            break
            
    print(f"\nTraining completed! Best Validation Macro F1 achieved: {best_macro_f1:.4f}")

if __name__ == "__main__":
    main()
