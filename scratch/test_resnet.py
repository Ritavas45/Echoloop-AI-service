import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights

print("Loading resnet18...")
try:
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()
    print("Success loading resnet18!")
except Exception as e:
    print(f"Error loading: {e}")
