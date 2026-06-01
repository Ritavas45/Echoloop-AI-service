import os
import shutil
import time
import torch
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
from database import ECholooopDataStore

def merge_datasets():
    """
    Combines the processed archive dataset (from data/archive_prepared)
    with the main data folder (data/train and data/val).
    """
    print("\n[Setup] Merging archive prepared datasets into main data/ directories...")
    src_train = "data/archive_prepared/train"
    src_val = "data/archive_prepared/val"
    dst_train = "data/train"
    dst_val = "data/val"
    
    classes = ['Reuse', 'Refurbish', 'Repair', 'Recycle']
    
    merged_count = 0
    for cls in classes:
        # Merge Train
        src_cls_train = os.path.join(src_train, cls)
        dst_cls_train = os.path.join(dst_train, cls)
        os.makedirs(dst_cls_train, exist_ok=True)
        if os.path.exists(src_cls_train):
            for item in os.listdir(src_cls_train):
                src_item = os.path.join(src_cls_train, item)
                dst_item = os.path.join(dst_cls_train, item)
                if os.path.isdir(src_item):
                    if os.path.exists(dst_item):
                        shutil.rmtree(dst_item)
                    shutil.copytree(src_item, dst_item)
                    merged_count += 1
                    
        # Merge Val
        src_cls_val = os.path.join(src_val, cls)
        dst_cls_val = os.path.join(dst_val, cls)
        os.makedirs(dst_cls_val, exist_ok=True)
        if os.path.exists(src_cls_val):
            for item in os.listdir(src_cls_val):
                src_item = os.path.join(src_cls_val, item)
                dst_item = os.path.join(dst_cls_val, item)
                if os.path.isdir(src_item):
                    if os.path.exists(dst_item):
                        shutil.rmtree(dst_item)
                    shutil.copytree(src_item, dst_item)
                    merged_count += 1
                    
    print(f"[Setup] ✓ Successfully merged {merged_count} phone folders into data/train and data/val.")

def print_confusion_matrix(cm, classes):
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
        avg_probs = model(images)
        log_probs = torch.log(avg_probs + 1e-15)
        loss = F.nll_loss(log_probs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds = avg_probs.argmax(dim=-1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
        
    return running_loss / total, correct / total

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
    
    macro_f1 = 0.0
    if SKLEARN_AVAILABLE:
        macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(classes))))
        print(f"\nValidation Classification Report:")
        print(classification_report(all_targets, all_preds, target_names=classes, zero_division=0))
        print("Confusion Matrix:")
        print_confusion_matrix(cm, classes)
    else:
        macro_f1 = val_acc
        
    return val_loss, val_acc, macro_f1

def main():
    # 1. Merge datasets
    merge_datasets()
    
    params = {
        'data_dir': './data',
        'epochs': 5,
        'batch_size': 16,
        'lr': 1e-3,
        'patience': 3,
        'save_dir': './checkpoints',
        'img_size': 300,
        'num_workers': 0
    }

    
    os.makedirs(params['save_dir'], exist_ok=True)
    
    # 2. Use stable CPU execution for maximum safety and NaN avoidance
    device = torch.device('cpu')
    print(f"\n[Training] Using device: {device} (CPU is chosen to completely bypass Apple Silicon GPU NaN bugs)")
    
    # 3. Transforms and datasets
    train_transform, val_transform = get_transforms(params['img_size'])
    train_dir = os.path.join(params['data_dir'], 'train')
    val_dir = os.path.join(params['data_dir'], 'val')
    
    train_dataset = PhoneConditionDataset(train_dir, transform=train_transform, max_images=1)
    val_dataset = PhoneConditionDataset(val_dir, transform=val_transform, max_images=1)

    
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=params['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=params['num_workers']
    )
    
    print(f"[Training] Data loaders created. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # 4. Initialize model
    classes = ['Reuse', 'Refurbish', 'Repair', 'Recycle']
    print('[Training] Initializing Late Fusion EfficientNet-B3 with frozen backbone...')
    model = LateFusionEfficientNet(num_classes=len(classes), pretrained=True)
    
    # Freeze core backbone parameters to speed up CPU training and avoid NaN weights
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Keep classifier head parameters trainable
    for param in model.backbone.classifier.parameters():
        param.requires_grad = True
        
    model = model.to(device)
    
    # We only pass trainable parameters to AdamW optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=params['lr'],
        weight_decay=1e-2
    )
    
    best_macro_f1 = -1.0
    
    print("\n[Training] Starting head-only transfer learning loop...")
    for epoch in range(1, params['epochs'] + 1):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, device, classes)
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:02d}/{params['epochs']} | Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100.2:.2f}% | Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100.0:.2f}% - Val Macro F1: {val_f1:.4f} | Time: {elapsed:.1f}s")
        
        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(params['save_dir'], 'best_model.pth'))
            print(f"==> Saved new best model checkpoint to ./checkpoints/best_model.pth (Macro F1: {best_macro_f1:.4f})")
            
    # Save final model
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(params['save_dir'], 'latest_model.pth'))
    print(f"\n[Training] Complete! Best Val Macro F1: {best_macro_f1:.4f}")
    
    # 5. Register in SQLite database
    print("\n[Database] Logging model to SQLite and marking it active...")
    store = ECholooopDataStore()
    
    model_version = f"v2_archive_train_{int(time.time())}"
    
    # Log model metadata
    success = store.log_model_metadata(
        model_version=model_version,
        model_type="efficientnet_b3",
        training_data_size=len(train_dataset),
        validation_accuracy=float(val_acc),
        test_accuracy=float(val_acc), # use val as proxy for test
        model_path="./checkpoints/best_model.pth",
        config={
            "backbone": "efficientnet_b3",
            "backbone_frozen": True,
            "epochs": params['epochs'],
            "batch_size": params['batch_size'],
            "lr": params['lr']
        },
        metrics={
            "val_accuracy": float(val_acc),
            "val_macro_f1": float(best_macro_f1)
        }
    )
    
    if success:
        store.set_active_model(model_version)
        print(f"[Database] ✓ Registered model version '{model_version}' and set it as active production model!")
    else:
        print("[Database] Warning: Failed to register model metadata.")

if __name__ == "__main__":
    main()
