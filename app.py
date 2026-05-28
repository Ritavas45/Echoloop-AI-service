import os
import io
from typing import List
from PIL import Image
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from dataset import get_transforms
from model import LateFusionEfficientNet

app = FastAPI(
    title="Multi-Modal Phone Condition Classifier",
    description="FastAPI endpoint fusing Late Fusion EfficientNet-B3 (images) and XGBoost (metadata) with confidence-gated voting.",
    version="1.0"
)

# 1. Globals for models and tools
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
image_model = None
tabular_model = None
val_transform = None
CLASSES = ['Reuse', 'Refurbish', 'Repair', 'Recycle']

# 2. Pydantic Models for Documented Output
class Probabilities(BaseModel):
    Reuse: float
    Refurbish: float
    Repair: float
    Recycle: float

class IndividualVotes(BaseModel):
    image_model_prediction: str
    image_model_confidence_pct: float
    tabular_model_prediction: str
    tabular_model_confidence_pct: float

class PredictionResponse(BaseModel):
    prediction: str
    confidence_pct: float
    fused_probability_breakdown: Probabilities
    individual_votes: IndividualVotes
    decision_path: str

@app.on_event("startup")
def startup_event():
    global image_model, tabular_model, val_transform
    print(f"Server starting. Target device for PyTorch: {device}")
    
    # Load transforms (we use 300x300 for EfficientNet-B3)
    _, val_transform = get_transforms(300)
    
    # Load Late Fusion PyTorch model
    print("Loading late-fusion image classifier backbone...")
    image_model = LateFusionEfficientNet(num_classes=len(CLASSES), pretrained=False)
    
    checkpoint_path = "./checkpoints/best_model.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Support loading state dictionary from standard saved checkpoint dict
            if 'model_state_dict' in checkpoint:
                image_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                image_model.load_state_dict(checkpoint)
            print("PyTorch model weights loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load state dict from {checkpoint_path}: {e}. Running with randomized weights.")
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Running with randomized weights.")
        
    image_model.to(device)
    image_model.eval()
    
    # Load XGBoost Tabular model
    print("Loading XGBoost tabular classifier...")
    xgb_path = "./xgboost_model.json"
    if os.path.exists(xgb_path):
        try:
            tabular_model = xgb.XGBClassifier()
            tabular_model.load_model(xgb_path)
            print("XGBoost model loaded successfully.")
        except Exception as e:
            print(f"Error loading XGBoost model: {e}")
            raise RuntimeError(f"Could not load tabular model: {e}")
    else:
        raise FileNotFoundError(f"Missing XGBoost model at {xgb_path}. Please run train_xgboost.py first.")

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    images: List[UploadFile] = File(...),
    model_age_months: int = Form(...),
    battery_health_pct: float = Form(...),
    screen_cracked: bool = Form(...),
    functional_issues: bool = Form(...),
    cosmetic_scratches: int = Form(...) # 0=none, 1=minor, 2=major
):
    # 1. Validate image count
    n_images = len(images)
    if n_images < 4 or n_images > 5:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid image count. You uploaded {n_images} images, but the model accepts exactly 4 or 5 images."
        )
        
    # 2. Process images
    image_tensors = []
    # Load each file into PIL and apply transforms
    for img_file in images:
        try:
            contents = await img_file.read()
            img = Image.open(io.BytesIO(contents)).convert('RGB')
            img_tensor = val_transform(img)
            image_tensors.append(img_tensor)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process image {img_file.filename}: {e}")
            
    # Pad to exactly 5 images if 4 images were uploaded (replicate last image)
    if len(image_tensors) == 4:
        image_tensors.append(image_tensors[-1])
        
    # Stack images: shape [5, C, H, W], and add batch dim: [1, 5, C, H, W]
    images_tensor = torch.stack(image_tensors, dim=0).unsqueeze(0).to(device)
    
    # 3. Image Model Inference
    with torch.no_grad():
        # Get fused average class probabilities: shape [1, 4]
        img_probs_tensor = image_model(images_tensor)
        img_probs = img_probs_tensor[0].cpu().numpy()
        
    # 4. Tabular Model Inference
    # Format tabular inputs to Pandas DataFrame with feature names to match XGBoost
    input_df = pd.DataFrame([{
        'model_age_months': model_age_months,
        'battery_health_pct': battery_health_pct,
        'screen_cracked': 1 if screen_cracked else 0,
        'functional_issues': 1 if functional_issues else 0,
        'cosmetic_scratches': cosmetic_scratches
    }])
    
    try:
        # Get probability distributions: shape [4]
        tab_probs = tabular_model.predict_proba(input_df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"XGBoost inference failed: {e}")
        
    # 5. Multi-Modal Decision Fusion
    # Model predictions
    img_pred_idx = int(np.argmax(img_probs))
    tab_pred_idx = int(np.argmax(tab_probs))
    
    img_max_conf = float(img_probs[img_pred_idx])
    tab_max_conf = float(tab_probs[tab_pred_idx])
    
    # Check for Confidence-Gated Override at 90% threshold (0.90)
    decision_path = "weighted_soft_voting"
    fused_probs = None
    
    if img_max_conf >= 0.90 and tab_max_conf < 0.90:
        fused_probs = img_probs
        decision_path = "image_model_override_90%"
    elif tab_max_conf >= 0.90 and img_max_conf < 0.90:
        fused_probs = tab_probs
        decision_path = "tabular_model_override_90%"
    elif img_max_conf >= 0.90 and tab_max_conf >= 0.90:
        # Both confident, pick the one with absolute highest confidence
        if img_max_conf >= tab_max_conf:
            fused_probs = img_probs
            decision_path = "image_model_override_90%"
        else:
            fused_probs = tab_probs
            decision_path = "tabular_model_override_90%"
    else:
        # Standard weighted soft voting: 60% image, 40% tabular
        fused_probs = 0.6 * img_probs + 0.4 * tab_probs
        decision_path = "weighted_soft_voting"
        
    fused_pred_idx = int(np.argmax(fused_probs))
    fused_conf = float(fused_probs[fused_pred_idx])
    
    # Convert numpy values to native Python floats/strings for JSON serialization
    fused_probabilities_dict = {CLASSES[i]: float(fused_probs[i]) for i in range(len(CLASSES))}
    
    # 6. Response Payload
    return PredictionResponse(
        prediction=CLASSES[fused_pred_idx],
        confidence_pct=fused_conf * 100.0,
        fused_probability_breakdown=Probabilities(**fused_probabilities_dict),
        individual_votes=IndividualVotes(
            image_model_prediction=CLASSES[img_pred_idx],
            image_model_confidence_pct=img_max_conf * 100.0,
            tabular_model_prediction=CLASSES[tab_pred_idx],
            tabular_model_confidence_pct=tab_max_conf * 100.0
        ),
        decision_path=decision_path
    )

@app.get("/")
def read_root():
    return {"status": "healthy", "service": "Mobile Phone Condition Classification API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
