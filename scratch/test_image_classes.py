import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torchvision.transforms as transforms
from PIL import Image

# Load model
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()

categories = weights.meta["categories"]

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_top_classes(image_path):
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    print(f"\nImage: {image_path}")
    for i in range(top5_prob.size(0)):
        idx = top5_catid[i].item()
        label = categories[idx]
        prob = top5_prob[i].item()
        print(f"  {i+1}: {label} (Index: {idx}, Prob: {prob:.4f})")

# Test on a few images if they exist
import os
base_path = "mock_data/val/Reuse/phone_001"
for file in sorted(os.listdir(base_path)):
    if file.endswith(".jpg"):
        predict_top_classes(os.path.join(base_path, file))
