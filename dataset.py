import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PhoneConditionDataset(Dataset):
    """
    Custom PyTorch Dataset for loading multi-image phone condition samples.
    
    Expected folder structure:
    root_dir/
      Reuse/
        sample_001/
          front.jpg
          back.jpg
          ...
      Refurbish/
        ...
      Repair/
        ...
      Recycle/
        ...
    """
    def __init__(self, root_dir, transform=None, max_images=5):
        """
        Args:
            root_dir (str): Path to root directory containing folders 'Reuse', 'Refurbish', 'Repair', 'Recycle'.
            transform (callable, optional): Transform to be applied to each individual image.
            max_images (int): Fixed number of images per phone sample folder.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.max_images = max_images
        
        self.classes = ['Reuse', 'Refurbish', 'Repair', 'Recycle']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        self._find_samples()
        
    def _find_samples(self):
        """
        Scans root_dir and compiles list of all phone folders and their associated images.
        """
        if not os.path.isdir(self.root_dir):
            print(f"Warning: Root directory {self.root_dir} does not exist.")
            return

        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            # Scan directory for folders (each represents a single phone sample)
            phone_folders = sorted(os.listdir(class_dir))
            for folder in phone_folders:
                folder_path = os.path.join(class_dir, folder)
                if not os.path.isdir(folder_path):
                    continue
                
                # Search for typical image files
                image_paths = []
                for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
                    image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
                
                # Ensure deterministic order of views
                image_paths = sorted(image_paths)
                
                # We only count this folder if it contains at least one image
                if len(image_paths) == 0:
                    continue
                    
                self.samples.append({
                    'folder_path': folder_path,
                    'image_paths': image_paths,
                    'label': self.class_to_idx[class_name]
                })
                
        print(f"Loaded {len(self.samples)} phone samples from {self.root_dir}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_paths = sample['image_paths']
        label = sample['label']
        
        n_images = len(image_paths)
        if n_images == 0:
            raise ValueError(f"No images found in sample folder: {sample['folder_path']}")
            
        # Replicate or truncate lists to fit exact self.max_images
        selected_paths = list(image_paths)
        if n_images < self.max_images:
            # Duplicate the last image to pad
            last_img = selected_paths[-1]
            while len(selected_paths) < self.max_images:
                selected_paths.append(last_img)
        elif n_images > self.max_images:
            # Truncate to maximum permitted images
            selected_paths = selected_paths[:self.max_images]
            
        image_tensors = []
        for path in selected_paths:
            try:
                img = Image.open(path).convert('RGB')
            except Exception as e:
                # Robust fallback for corrupted images
                print(f"Warning: Corrupt or missing image at {path}. Using zero tensor. Error: {e}")
                img = Image.new('RGB', (300, 300), color=0)
                
            if self.transform:
                img_tensor = self.transform(img)
            else:
                img_tensor = transforms.ToTensor()(img)
                
            image_tensors.append(img_tensor)
            
        # Stack images: shape [max_images, C, H, W]
        images_tensor = torch.stack(image_tensors, dim=0)
        
        return images_tensor, label

def get_transforms(img_size=300):
    """
    Standard pre-configured PyTorch image transforms for EfficientNet-B3.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform
