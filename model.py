import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class LateFusionEfficientNet(nn.Module):
    """
    Late Fusion EfficientNet-B3 model.
    It takes a batch of multi-image inputs, runs each image through the EfficientNet backbone,
    applies softmax to get individual probability distributions, and averages them to form
    a final class probability prediction for each phone sample.
    """
    def __init__(self, num_classes=4, pretrained=True):
        super(LateFusionEfficientNet, self).__init__()
        self.num_classes = num_classes
        
        # Load pre-trained EfficientNet-B3 model
        # Supporting both older (pretrained=True) and newer (weights=...) torchvision APIs
        try:
            from torchvision.models import EfficientNet_B3_Weights
            weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b3(weights=weights)
        except ImportError:
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            
        # EfficientNet-B3 features output dimension is 1536
        in_features = self.backbone.classifier[1].in_features
        
        # Replace the final linear layer with one matching the target class count (4)
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tensor of shape [BatchSize, NumImages, C, H, W]
            
        Returns:
            torch.Tensor: Averaged class probabilities of shape [BatchSize, NumClasses]
        """
        # x shape: [B, N, C, H, W]
        batch_size, num_images, c, h, w = x.shape
        
        # Flatten batch and multi-image dimensions to process in parallel
        # Flat shape: [B * N, C, H, W]
        x_flat = x.view(batch_size * num_images, c, h, w)
        
        # Extract logits from backbone
        # Output shape: [B * N, NumClasses]
        logits_flat = self.backbone(x_flat)
        
        # Reshape logits back to group by sample
        # Shape: [B, N, NumClasses]
        logits = logits_flat.view(batch_size, num_images, self.num_classes)
        
        # Apply softmax across classes for each individual image prediction
        # Shape: [B, N, NumClasses]
        probs = F.softmax(logits, dim=-1)
        
        # Perform late fusion: average softmax probabilities over the N images
        # Shape: [B, NumClasses]
        avg_probs = probs.mean(dim=1)
        
        return avg_probs

if __name__ == "__main__":
    # Quick shape verification code
    print("Testing LateFusionEfficientNet model...")
    model = LateFusionEfficientNet(num_classes=4, pretrained=False)
    
    # Dummy batch: 2 phone samples, each having 5 images of size 3x300x300
    dummy_input = torch.randn(2, 5, 3, 300, 300)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Softmax probability sums (should be 1.0):", output.sum(dim=-1).tolist())
    print("Model loaded and tested successfully!")
