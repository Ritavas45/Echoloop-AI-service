import torchvision.models as models
from torchvision.models import ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
categories = weights.meta["categories"]

keywords = ["phone", "telephone", "computer", "camera", "screen", "calculator", "device", "ipod", "monitor"]
for i, cat in enumerate(categories):
    cat_lower = cat.lower()
    for kw in keywords:
        if kw in cat_lower:
            print(f"Index: {i}, Label: '{cat}'")
            break
