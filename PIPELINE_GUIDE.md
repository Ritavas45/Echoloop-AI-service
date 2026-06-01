"""
Comprehensive Pipeline Documentation and System Architecture
"""

# ============================================================================
# SYSTEM ARCHITECTURE OVERVIEW
# ============================================================================

ARCHITECTURE_DOCS = """
# Echoloop AI - Continuous Learning Pipeline Architecture

## System Overview

The system implements a **Retrieval-Augmented Generation (RAG)** approach for continuous model improvement:

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERACTION                         │
│  ┌──────────────┐        ┌──────────────┐      ┌─────────────┐ │
│  │   Upload     │        │   Receive    │      │   Submit    │ │
│  │   Images +   │──────> │  Prediction  │────> │  Feedback   │ │
│  │   Features   │        │  + ID        │      │  (Label)    │ │
│  └──────────────┘        └──────────────┘      └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION ENGINE                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Multi-Modal Fusion (Late Fusion)                        │   │
│  │  ┌──────────────────┐          ┌──────────────────────┐ │   │
│  │  │ Image Model      │ 60%      │ Tabular Model        │ │   │
│  │  │ EfficientNet-B3  ├────┐     │ XGBoost              │ │   │
│  │  │ (5 Images)       │    ├───> │ (5 Features)         │ │   │
│  │  └──────────────────┘    │     └──────────────────────┘ │   │
│  │                          │            40%               │   │
│  │  Confidence-Gated Override (90% threshold)             │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION (RAG Store)                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Database: predictions_log + feedback                     │   │
│  │ - Prediction details (images, features, confidence)      │   │
│  │ - Ground truth labels (user corrections)                 │   │
│  │ - Metadata (timestamps, decision paths)                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 CONTINUOUS TRAINING PIPELINE                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Automatic Triggers:                                     │   │
│  │  - When threshold_samples (e.g., 50) new labels received │   │
│  │  - Scheduled (daily at 2 AM)                             │   │
│  │  - Manual via /retrain endpoint                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Process:                                                │   │
│  │  1. Fetch all labeled data from database                 │   │
│  │  2. Split: 70% train, 10% val, 20% test                 │   │
│  │  3. Retrain XGBoost on new data                          │   │
│  │  4. Evaluate on test set                                 │   │
│  │  5. Save new model with version tag                      │   │
│  │  6. Log metrics to database                              │   │
│  │  7. Deploy if meets criteria                             │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              MODEL VERSIONING & DEPLOYMENT                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Model Registry:                                         │   │
│  │  - v1_base (initial model)                               │   │
│  │  - v1_retrain_20240601_120000_abc12345                   │   │
│  │  - v1_retrain_20240602_020000_def67890 ◄─ ACTIVE        │   │
│  │                                                           │   │
│  │  Only ONE model is ACTIVE at any time                    │   │
│  │  Automatic A/B testing capability                        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. **database.py - Data Store (RAG Backend)**
   - **Purpose**: Central repository for all training data and feedback
   - **Tables**:
     * `predictions_log`: Every prediction with input features, outputs, confidence
     * `model_metadata`: Model versions, accuracy, training parameters
     * `training_jobs`: Retraining job history and status
     * `class_distribution`: Monitors data imbalance over time
   - **Key Methods**:
     * `log_prediction()`: Store new predictions
     * `submit_feedback()`: Record user corrections (ground truth)
     * `get_training_data()`: Fetch labeled data for retraining
     * `should_retrain()`: Check if threshold reached

### 2. **app_updated.py - FastAPI Application**
   - **Endpoints**:
     * `POST /predict`: Make predictions and log them
     * `POST /feedback`: Submit ground truth labels
     * `GET /model/status`: Current model performance
     * `POST /retrain`: Manually trigger retraining
     * `GET /predictions/pending-feedback`: Get samples needing labels
     * `GET /statistics`: Data collection metrics
   - **Features**:
     * Automatic background retraining when threshold reached
     * Prediction logging for future training
     * Feedback collection mechanism

### 3. **continuous_training.py - Training Pipeline**
   - **ContinuousTrainer Class**:
     * `prepare_training_data()`: Fetch and split labeled data
     * `train_tabular_model()`: Retrain XGBoost
     * `retrain_pipeline()`: Full retraining workflow
     * `compare_models()`: Compare model versions
   - **Versioning**: Unique model identifiers with timestamps
   - **Evaluation**: Automatic train/val/test metrics

### 4. **orchestrator.py - Pipeline Management**
   - **PipelineOrchestrator Class**:
     * Manages scheduled retraining jobs
     * Monitors data distribution for imbalance
     * Decides model deployment
     * Generates reports
   - **Features**:
     * Configurable schedule (daily, hourly, etc.)
     * Auto-deploy when criteria met
     * Alert on data imbalance
     * CLI interface for manual commands

## Data Flow: Continuous Learning Cycle

```
DAY 1-4: COLLECTION PHASE
├─ User uploads phone images + features
├─ Model makes prediction
├─ Prediction logged in database (with ID)
└─ Awaiting user feedback

DAY 5: USER PROVIDES FEEDBACK
├─ User submits ground truth label
├─ Database updates prediction_log
├─ Counter: 50+ new labels received ✓
└─ Threshold reached → Retrain triggered

DAY 5 02:00 AM: RETRAINING PHASE
├─ Orchestrator starts scheduled job
├─ Fetch all labeled data from database
├─ Split: 70% train, 10% val, 20% test
├─ Train new XGBoost model
├─ Evaluate: 
│  ├─ Train Acc: 89%
│  ├─ Val Acc: 87%
│  ├─ Test Acc: 86%
├─ Save model with version: v1_retrain_20240605_020000_xyz789
├─ Log metrics to database
└─ Check deployment criteria

DAY 5 02:15 AM: DEPLOYMENT DECISION
├─ Is Test Acc ≥ 80%? ✓ YES (86%)
├─ Is accuracy drop ≤ 3%? ✓ YES (previous: 87%)
├─ Auto-deploy enabled? ✓ YES
└─ Set new model as ACTIVE ✓

DAY 6+: NEW PREDICTIONS USE UPDATED MODEL
└─ All predictions now use v1_retrain_20240605_020000_xyz789
```

## Database Schema

### predictions_log Table
```sql
CREATE TABLE predictions_log (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,              -- When prediction was made
    model_version TEXT,              -- Which model made this prediction
    
    -- Input Features
    model_age_months INTEGER,
    battery_health_pct REAL,
    screen_cracked INTEGER,
    functional_issues INTEGER,
    cosmetic_scratches INTEGER,
    image_filenames TEXT,            -- JSON list of uploaded images
    
    -- Model Outputs
    image_pred TEXT,                 -- Image model prediction
    image_confidence REAL,
    tabular_pred TEXT,               -- Tabular model prediction
    tabular_confidence REAL,
    fused_prediction TEXT,           -- Final ensemble prediction
    fused_confidence REAL,
    decision_path TEXT,              -- Which fusion rule was used
    raw_probabilities TEXT,          -- JSON with all class probabilities
    
    -- User Feedback (Ground Truth) - Filled Later
    feedback_received INTEGER,       -- 0=pending, 1=received
    ground_truth TEXT,               -- User's correct label
    feedback_timestamp DATETIME,     -- When user submitted feedback
    confidence_correct INTEGER,      -- Was model's confidence correct?
    
    -- Tracking
    is_training_data INTEGER         -- 0=not used in training yet, 1=used
);
```

## Deployment Architecture

### Local Development
```
docker-compose up
├─ echoloop-api: REST API on :8000
└─ orchestrator: Background scheduler for retraining
```

### Production (Kubernetes)
```
echoloop-api-deployment (3 replicas)
├─ Auto-scales based on CPU/memory
├─ Rolling updates for zero-downtime deployments
├─ Health checks every 30 seconds
└─ LoadBalancer service routes traffic

echoloop-retrain (CronJob)
├─ Runs daily at 2 AM
├─ Triggers retraining if data available
├─ Mounts shared PVC for models/data
└─ Auto-deploy to production if criteria met

Persistent Volumes
├─ data-pvc: 100GB for SQLite database + logs
└─ models-pvc: 50GB for model checkpoints
```

## Configuration

### pipeline_config.json
```json
{
  "retraining": {
    "enabled": true,
    "threshold_samples": 50,
    "schedule": "daily",
    "time": "02:00"
  },
  "deployment": {
    "auto_deploy": false,
    "min_test_accuracy": 0.80,
    "max_performance_drop": 0.03
  },
  "data_collection": {
    "retention_days": 365,
    "alert_on_imbalance": true,
    "max_class_ratio": 3.0
  }
}
```

## Monitoring & Observability

### Key Metrics
- Prediction count (daily, weekly, monthly)
- Feedback rate (% of predictions labeled)
- Model accuracy (train/val/test)
- Class distribution (detect imbalance)
- Retraining frequency and success rate
- Inference latency (p50, p95, p99)
- Data collection rate

### Alerts
- ⚠ Data imbalance > 3:1 ratio
- ⚠ Prediction accuracy drop > 5%
- ⚠ Feedback rate < 10%
- ⚠ Retrain job failure
- ⚠ New model rejects to deploy
"""


# ============================================================================
# GETTING STARTED GUIDE
# ============================================================================

GETTING_STARTED = """
# Getting Started with Echoloop AI Continuous Learning System

## Quick Start (Development)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install schedule uvicorn  # Additional for orchestrator
```

### 2. Initialize Database
```bash
python -c "from database import ECholooopDataStore; ECholooopDataStore()"
```

### 3. Run the API
```bash
python -m uvicorn app_updated:app --reload --host 0.0.0.0 --port 8000
```

### 4. Start the Orchestrator (in another terminal)
```bash
python orchestrator.py scheduler
```

### 5. Make a Prediction
```bash
curl -X POST "http://localhost:8000/predict" \\
  -F "images=@phone1.jpg" \\
  -F "images=@phone2.jpg" \\
  -F "images=@phone3.jpg" \\
  -F "images=@phone4.jpg" \\
  -F "model_age_months=24" \\
  -F "battery_health_pct=75" \\
  -F "screen_cracked=false" \\
  -F "functional_issues=false" \\
  -F "cosmetic_scratches=2"
```

Response:
```json
{
  "prediction_id": 1,
  "prediction": "Refurbish",
  "confidence_pct": 82.5,
  "model_version": "v1_base",
  ...
}
```

### 6. Submit Feedback
```bash
curl -X POST "http://localhost:8000/feedback" \\
  -H "Content-Type: application/json" \\
  -d '{
    "prediction_id": 1,
    "ground_truth": "Refurbish",
    "user_confidence": 0.95
  }'
```

### 7. Monitor Retraining Status
```bash
curl http://localhost:8000/model/status
```

## Production Deployment

### Docker Deployment
```bash
# Build image
docker build -t echoloop:latest .

# Run with Docker Compose
docker-compose up -d
```

### Kubernetes Deployment
```bash
# Create namespace
kubectl create namespace echoloop

# Deploy
kubectl apply -f deployment/k8s/deployment.yaml -n echoloop

# Check status
kubectl get deployments -n echoloop
kubectl get pods -n echoloop
```

## Managing the Pipeline

### Command-Line Interface

#### Trigger Manual Retraining
```bash
python orchestrator.py retrain
```

#### Monitor Data Distribution
```bash
python orchestrator.py monitor
```

#### Generate Report
```bash
python orchestrator.py report
```

#### View Configuration
```bash
python orchestrator.py config --show
```

#### Reset Configuration
```bash
python orchestrator.py config --reset
```

#### Start Continuous Scheduler
```bash
python orchestrator.py scheduler
```

## Best Practices

### 1. Data Collection
- Aim for at least 50 labeled samples before retraining
- Ensure balanced class distribution (max 3:1 ratio)
- Collect feedback regularly (target: >50% feedback rate)

### 2. Model Versioning
- Keep all model versions (enables rollback)
- Track metrics for each version
- Use semantic versioning: v1_retrain_YYYYMMDD_HHMMSS_uniqueid

### 3. Deployment
- Set reasonable thresholds:
  * Minimum test accuracy: 80%
  * Max accuracy drop: 3%
- Test on a representative dataset before deployment
- Monitor production model performance

### 4. Monitoring
- Check class distribution weekly
- Alert on data imbalance
- Track feedback rate
- Monitor inference latency

## Troubleshooting

### Issue: Retraining not triggered
**Solution**: 
```bash
# Check if threshold is reached
curl http://localhost:8000/predictions/pending-feedback

# Manually trigger
python orchestrator.py retrain
```

### Issue: Model accuracy dropped after retraining
**Solution**:
```bash
# Check class distribution
python orchestrator.py monitor

# Verify the new model wasn't deployed (if auto_deploy=false)
curl http://localhost:8000/model/status
```

### Issue: Database grows too large
**Solution**:
- Configure retention policy in pipeline_config.json
- Archive old predictions to separate storage
- Implement data cleanup script

## API Reference

### POST /predict
Make a phone condition prediction.

**Request**:
- images: 4-5 image files
- model_age_months: integer
- battery_health_pct: float (0-100)
- screen_cracked: boolean
- functional_issues: boolean
- cosmetic_scratches: integer (0-10)

**Response**:
- prediction_id: integer (for later feedback)
- prediction: string (Reuse/Refurbish/Repair/Recycle)
- confidence_pct: float
- fused_probability_breakdown: probabilities for all classes
- individual_votes: image and tabular model votes
- model_version: which model made this prediction

### POST /feedback
Submit ground truth label for a prediction.

**Request**:
- prediction_id: integer
- ground_truth: string (Reuse/Refurbish/Repair/Recycle)
- user_confidence: float (optional, 0-1)

### GET /model/status
Get current model performance and retraining status.

### POST /retrain
Manually trigger model retraining.

### GET /statistics
Get data collection and class distribution statistics.

### GET /health
Health check endpoint.
"""


if __name__ == "__main__":
    print(ARCHITECTURE_DOCS)
    print("\n" + "="*80 + "\n")
    print(GETTING_STARTED)
