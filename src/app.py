"""
Updated FastAPI application with continuous learning integration.
Includes data collection, feedback endpoints, and model versioning.
"""

import xgboost as xgb
import os
import io
from typing import List, Optional
from datetime import datetime
from PIL import Image
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel
import json

from dataset import get_transforms
from model import LateFusionEfficientNet
from database import ECholooopDataStore
from continuous_training import ContinuousTrainer, check_and_retrain_if_needed
from gcp_storage import GCPStorageManager

# ============================================================================
# Initialize FastAPI App
# ============================================================================

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Echoloop AI - Multi-Modal Phone Condition Classifier with Continuous Learning",
    description="FastAPI endpoint with Late Fusion EfficientNet-B3 + XGBoost + Continuous Training Pipeline",
    version="2.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Configuration
# ============================================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

CLASSES = ['Reuse', 'Refurbish', 'Repair', 'Recycle']
RETRAINING_THRESHOLD = 50  # Retrain after 50 new labels
CURRENT_MODEL_VERSION = "v2_archive_train_epoch1"


# Global models and data store
image_model = None
tabular_model = None
val_transform = None
data_store = None
trainer = None

# ============================================================================
# Pydantic Models
# ============================================================================

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
    prediction_id: int
    prediction: str
    confidence_pct: float
    fused_probability_breakdown: Probabilities
    individual_votes: IndividualVotes
    decision_path: str
    model_version: str
    timestamp: str

class FeedbackRequest(BaseModel):
    prediction_id: int
    ground_truth: str  # One of ['Reuse', 'Refurbish', 'Repair', 'Recycle']
    user_confidence: Optional[float] = None

class FeedbackResponse(BaseModel):
    success: bool
    message: str
    prediction_id: int

class ModelStatus(BaseModel):
    active_model_version: str
    accuracy_val: float
    accuracy_test: float
    training_samples: int
    last_retrain: Optional[str]
    retraining_required: bool

class RetrainingStatus(BaseModel):
    success: bool
    message: str
    new_model_version: Optional[str] = None
    model_accuracy: Optional[float] = None

# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
def startup_event():
    """Initialize models and data store on app startup."""
    global image_model, tabular_model, val_transform, data_store, trainer
    
    print(f"\n{'='*60}")
    print(f"[Echoloop] Server starting on device: {device}")
    print(f"{'='*60}\n")
    
    # Initialize data store
    print("[Echoloop] Initializing data store...")
    data_store = ECholooopDataStore()
    
    # Sync checkpoints from GCP Storage if configured
    try:
        storage_mgr = GCPStorageManager()
        if storage_mgr.enabled:
            storage_mgr.sync_checkpoints_from_gcs()
    except Exception as e:
        print(f"[Echoloop] Warning: Could not sync from GCS on startup: {e}")

    trainer = ContinuousTrainer(device=device, data_store=data_store)
    
    # Load transforms
    print("[Echoloop] Loading image transforms...")
    _, val_transform = get_transforms(300)
    
    # Load tabular model (XGBoost) first to avoid library/OpenMP conflict segfaults on macOS
    print("[Echoloop] Loading XGBoost tabular classifier...")
    xgb_path = "./xgboost_model.json"
    if os.path.exists(xgb_path):
        try:
            tabular_model = xgb.XGBClassifier()
            tabular_model.load_model(xgb_path)
            print("[Echoloop] ✓ XGBoost model loaded successfully")
        except Exception as e:
            print(f"[Echoloop] Error: Could not load XGBoost model: {e}")
            raise RuntimeError(f"Could not load tabular model: {e}")
    else:
        raise FileNotFoundError(f"Missing XGBoost model at {xgb_path}")

    # Load image model (EfficientNet)
    print("[Echoloop] Loading Late Fusion EfficientNet-B3...")
    image_model = LateFusionEfficientNet(num_classes=len(CLASSES), pretrained=False)
    
    checkpoint_path = "./checkpoints/best_model.pth"
    if os.path.exists(checkpoint_path):
        print(f"[Echoloop] Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                image_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                image_model.load_state_dict(checkpoint)
            print("[Echoloop] ✓ Image model loaded successfully")
        except Exception as e:
            print(f"[Echoloop] Warning: Failed to load checkpoint: {e}")
    else:
        print(f"[Echoloop] Warning: Checkpoint not found. Using random weights.")
    
    image_model.to(device)
    image_model.eval()
    
    # Start the orchestrator scheduler in a background thread
    try:
        from orchestrator import PipelineOrchestrator
        import threading
        
        def run_scheduler():
            print("[Echoloop] Starting background orchestrator scheduler thread...")
            orchestrator = PipelineOrchestrator()
            orchestrator.start_scheduler()
            
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        print("[Echoloop] ✓ Orchestrator scheduler thread started successfully")
    except Exception as e:
        print(f"[Echoloop] ⚠️ Failed to start orchestrator scheduler thread: {e}")
        
    print(f"\n{'='*60}")
    print(f"[Echoloop] All models loaded and ready!")
    print(f"{'='*60}\n")

@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on shutdown."""
    print("\n[Echoloop] Server shutting down...")

# ============================================================================
# API Endpoints: Prediction
# ============================================================================

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    images: List[UploadFile] = File(..., description="Upload 4 or 5 phone images"),
    model_age_months: int = Form(...),
    battery_health_pct: float = Form(...),
    screen_cracked: bool = Form(...),
    functional_issues: bool = Form(...),
    cosmetic_scratches: int = Form(...)
):
    """
    Make a prediction on phone condition.
    Combines image and tabular features using late fusion.
    Logs prediction for continuous learning.
    """
    
    # 1. Validate image count
    n_images = len(images)
    if n_images < 4 or n_images > 5:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image count: {n_images}. Expected 4-5 images."
        )
    
    # 2. Process images
    image_tensors = []
    image_filenames = []
    
    for img_file in images:
        try:
            contents = await img_file.read()
            img = Image.open(io.BytesIO(contents)).convert('RGB')
            img_tensor = val_transform(img)
            image_tensors.append(img_tensor)
            image_filenames.append(img_file.filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process {img_file.filename}: {e}")
    
    # Pad to 5 images if needed
    if len(image_tensors) == 4:
        image_tensors.append(image_tensors[-1])
        image_filenames.append(f"{image_filenames[-1]}_padded")
    
    images_tensor = torch.stack(image_tensors, dim=0).unsqueeze(0).to(device)
    
    # 3. Image Model Inference
    with torch.no_grad():
        img_probs_tensor = image_model(images_tensor)
        img_probs = img_probs_tensor[0].cpu().numpy()
    
    # 4. Tabular Model Inference
    input_df = pd.DataFrame([{
        'model_age_months': model_age_months,
        'battery_health_pct': battery_health_pct,
        'screen_cracked': int(screen_cracked),
        'functional_issues': int(functional_issues),
        'cosmetic_scratches': cosmetic_scratches
    }])
    
    try:
        tab_probs = tabular_model.predict_proba(input_df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tabular inference failed: {e}")
    
    # 5. Multi-Modal Decision Fusion (Confidence-Gated Voting)
    img_pred_idx = int(np.argmax(img_probs))
    tab_pred_idx = int(np.argmax(tab_probs))
    
    img_max_conf = float(img_probs[img_pred_idx])
    tab_max_conf = float(tab_probs[tab_pred_idx])
    
    decision_path = "weighted_soft_voting"
    fused_probs = None
    
    if img_max_conf >= 0.90 and tab_max_conf < 0.90:
        fused_probs = img_probs
        decision_path = "image_model_override_90%"
    elif tab_max_conf >= 0.90 and img_max_conf < 0.90:
        fused_probs = tab_probs
        decision_path = "tabular_model_override_90%"
    elif img_max_conf >= 0.90 and tab_max_conf >= 0.90:
        if img_max_conf >= tab_max_conf:
            fused_probs = img_probs
            decision_path = "image_model_override_90%"
        else:
            fused_probs = tab_probs
            decision_path = "tabular_model_override_90%"
    else:
        # Weighted soft voting: 60% image, 40% tabular
        fused_probs = 0.6 * img_probs + 0.4 * tab_probs
        decision_path = "weighted_soft_voting"
    
    fused_pred_idx = int(np.argmax(fused_probs))
    fused_conf = float(fused_probs[fused_pred_idx])
    
    # 6. Log prediction to database
    prediction_id = data_store.log_prediction(
        model_version=CURRENT_MODEL_VERSION,
        model_age_months=model_age_months,
        battery_health_pct=battery_health_pct,
        screen_cracked=screen_cracked,
        functional_issues=functional_issues,
        cosmetic_scratches=cosmetic_scratches,
        image_pred=CLASSES[img_pred_idx],
        image_confidence=img_max_conf,
        tabular_pred=CLASSES[tab_pred_idx],
        tabular_confidence=tab_max_conf,
        fused_prediction=CLASSES[fused_pred_idx],
        fused_confidence=fused_conf,
        decision_path=decision_path,
        raw_probabilities={c: float(fused_probs[i]) for i, c in enumerate(CLASSES)},
        image_filenames=image_filenames
    )
    
    print(f"[Echoloop] Prediction logged with ID: {prediction_id}")
    
    # 7. Return response
    fused_probabilities_dict = {
        CLASSES[i]: float(fused_probs[i]) for i in range(len(CLASSES))
    }
    
    return PredictionResponse(
        prediction_id=prediction_id,
        prediction=CLASSES[fused_pred_idx],
        confidence_pct=fused_conf * 100.0,
        fused_probability_breakdown=Probabilities(**fused_probabilities_dict),
        individual_votes=IndividualVotes(
            image_model_prediction=CLASSES[img_pred_idx],
            image_model_confidence_pct=img_max_conf * 100.0,
            tabular_model_prediction=CLASSES[tab_pred_idx],
            tabular_model_confidence_pct=tab_max_conf * 100.0
        ),
        decision_path=decision_path,
        model_version=CURRENT_MODEL_VERSION,
        timestamp=datetime.now().isoformat()
    )

# ============================================================================
# API Endpoints: Feedback & Continuous Learning
# ============================================================================

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    Submit ground truth feedback for a prediction.
    Triggers retraining pipeline if threshold is reached.
    """
    
    # Validate ground truth
    if feedback.ground_truth not in CLASSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ground_truth. Must be one of {CLASSES}"
        )
    
    # Submit feedback
    success = data_store.submit_feedback(
        prediction_id=feedback.prediction_id,
        ground_truth=feedback.ground_truth,
        correct=True  # You can enhance this based on user confidence
    )
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction with ID {feedback.prediction_id} not found"
        )
    
    print(f"[Echoloop] Feedback received for prediction {feedback.prediction_id}: {feedback.ground_truth}")
    
    # Check if retraining is needed (run in background)
    if data_store.should_retrain(threshold_samples=RETRAINING_THRESHOLD):
        print(f"[Echoloop] Triggering retraining pipeline in background...")
        background_tasks.add_task(check_and_retrain_if_needed, RETRAINING_THRESHOLD)
    
    return FeedbackResponse(
        success=True,
        message=f"Feedback recorded for prediction {feedback.prediction_id}",
        prediction_id=feedback.prediction_id
    )

@app.get("/model/status", response_model=ModelStatus)
async def get_model_status():
    """Get current model status and training metrics."""
    
    active_model = data_store.get_active_model()
    
    if not active_model:
        raise HTTPException(status_code=404, detail="No active model found")
    
    labeled_data = len(data_store.get_unfeedback_predictions(limit=999999))
    retraining_required = data_store.should_retrain(RETRAINING_THRESHOLD)
    
    return ModelStatus(
        active_model_version=active_model['model_version'],
        accuracy_val=active_model['validation_accuracy'],
        accuracy_test=active_model['test_accuracy'],
        training_samples=active_model['training_data_size'],
        last_retrain=active_model['created_date'],
        retraining_required=retraining_required
    )

@app.post("/retrain", response_model=RetrainingStatus)
async def trigger_retraining(background_tasks: BackgroundTasks):
    """
    Manually trigger model retraining with collected feedback data.
    Runs in background for long-running operations.
    """
    
    # Check if enough data
    labeled_count = len(data_store.get_unfeedback_predictions(limit=999999))
    
    if labeled_count < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough labeled data for retraining. Need at least 10, have {labeled_count}"
        )
    
    # Run retraining in background
    def retrain_task():
        result = check_and_retrain_if_needed(RETRAINING_THRESHOLD)
        if result and result['success']:
            print(f"[Echoloop] ✓ Retraining completed: {result['model_version']}")
            print(f"[Echoloop] Test Accuracy: {result['metrics']['test_accuracy']:.4f}")
    
    background_tasks.add_task(retrain_task)
    
    return RetrainingStatus(
        success=True,
        message="Retraining pipeline started in background..."
    )

@app.get("/predictions/pending-feedback")
async def get_pending_feedback(limit: int = 20):
    """Get predictions awaiting user feedback."""
    df = data_store.get_unfeedback_predictions(limit=limit)
    
    if len(df) == 0:
        return {"count": 0, "predictions": []}
    
    predictions = []
    for _, row in df.iterrows():
        predictions.append({
            "prediction_id": row['id'],
            "timestamp": row['timestamp'],
            "fused_prediction": row['fused_prediction'],
            "confidence": row['fused_confidence'],
            "decision_path": row['decision_path']
        })
    
    return {"count": len(predictions), "predictions": predictions}

@app.get("/statistics")
async def get_statistics():
    """Get data collection and model statistics."""
    stats = trainer.get_training_statistics()
    return stats

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": str(device),
        "model_version": CURRENT_MODEL_VERSION,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# Additional Utility Endpoints
# ============================================================================

@app.get("/models/versions")
async def get_model_versions():
    """Get all available model versions."""
    # This would query the database for all models
    # For now, return a placeholder
    return {"message": "Model version history not yet implemented"}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
