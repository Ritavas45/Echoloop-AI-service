# Echoloop AI - Continuous Learning Pipeline

A complete machine learning system for **phone condition classification** with **continuous learning capabilities**. The system collects user feedback and automatically retrains models for continuous improvement.

## 🎯 System Overview

This system implements a **Retrieval-Augmented Generation (RAG) approach** for continuous model training:

```
User Predictions → Data Store (RAG Backend) → Feedback Collection → Automatic Retraining → Better Models
```

### Key Features

- ✅ **Multi-Modal Predictions**: Combines image (EfficientNet-B3) and tabular (XGBoost) models
- ✅ **Continuous Learning**: Automatically retrains models with user feedback
- ✅ **Data Collection**: Centralized database stores all predictions and feedback
- ✅ **Model Versioning**: Track and manage multiple model versions
- ✅ **Automated Deployment**: Auto-deploy when criteria are met
- ✅ **Scheduled Retraining**: Configurable retraining schedules
- ✅ **Production Ready**: Docker & Kubernetes deployment support
- ✅ **Comprehensive Monitoring**: Track model performance and data distribution

## 📊 System Architecture

```
┌─────────────────────────────────────────────────┐
│     FastAPI (app_updated.py)                    │
│  - Prediction endpoint                          │
│  - Feedback collection                          │
│  - Model status monitoring                      │
└────────────────────┬────────────────────────────┘
                     │
         ┌───────────┼────────────┐
         ▼           ▼            ▼
    ┌─────────┐ ┌────────┐  ┌──────────┐
    │ Database│ │Image   │  │Tabular   │
    │(RAG)    │ │Model   │  │Model     │
    │SQLite   │ │Eff-Net │  │XGBoost   │
    └────┬────┘ └────────┘  └──────────┘
         │
         ▼
    ┌─────────────────────────────┐
    │ ContinuousTrainer           │
    │ (continuous_training.py)    │
    │ - Retrain logic             │
    │ - Model versioning          │
    └────────────┬────────────────┘
         │
         ▼
    ┌──────────────────────────────┐
    │ PipelineOrchestrator         │
    │ (orchestrator.py)            │
    │ - Scheduling                 │
    │ - Monitoring                 │
    │ - Deployment decisions       │
    └──────────────────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to project
cd /path/to/Echoloop-AI-service

# Run full setup (interactive)
bash setup.sh

# Or setup with a command
bash setup.sh setup
```

This will:
- Create necessary directories
- Set up Python virtual environment
- Install dependencies
- Initialize database
- Generate deployment configs

### 2. Start the API Server

**Terminal 1: Start API**
```bash
bash setup.sh api
```

or manually:
```bash
source venv/bin/activate
python -m uvicorn app_updated:app --host 0.0.0.0 --port 8000
```

### 3. Start the Orchestrator (Background Scheduler)

**Terminal 2: Start Orchestrator**
```bash
bash setup.sh orchestrator
```

or manually:
```bash
source venv/bin/activate
python orchestrator.py scheduler
```

### 4. Test the System

```bash
# Test API health
curl http://localhost:8000/health

# Make a prediction
python client.py
```

## 📝 Usage Examples

### Making a Prediction

```python
from client import EcholooopClient

client = EcholooopClient("http://localhost:8000")

result = client.predict(
    image_paths=[
        "phone_image_1.jpg",
        "phone_image_2.jpg",
        "phone_image_3.jpg",
        "phone_image_4.jpg"
    ],
    model_age_months=24,
    battery_health_pct=75.0,
    screen_cracked=False,
    functional_issues=False,
    cosmetic_scratches=2
)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence_pct']:.1f}%")
print(f"Prediction ID: {result['prediction_id']}")
```

### Submitting Feedback (Ground Truth)

```python
# Submit the correct label for model improvement
client.submit_feedback(
    prediction_id=result['prediction_id'],
    ground_truth="Refurbish",  # Correct label
    user_confidence=0.95
)

print("Feedback submitted - model will retrain with this data!")
```

### Checking Model Status

```python
status = client.get_model_status()
print(f"Active Model: {status['active_model_version']}")
print(f"Test Accuracy: {status['accuracy_test']:.2%}")
print(f"Retrain Needed: {status['retraining_required']}")
```

### Getting Statistics

```python
stats = client.get_statistics()
print("Class Distribution (Last 7 days):")
print(stats['data_collection']['class_distribution_7d'])
```

## 🔄 Continuous Learning Workflow

### How the System Works

1. **Prediction Phase** (Minutes 0-5)
   - User uploads 4-5 phone images + device features
   - Model makes prediction and returns confidence
   - Prediction logged in database with unique ID

2. **Feedback Collection Phase** (Hours-Days)
   - User verifies prediction and submits correct label
   - System stores ground truth in database
   - Counter increments for each new label

3. **Retraining Trigger** (Automatic)
   - When 50+ new labels collected → Retrain triggered
   - Or scheduled daily at 2 AM (configurable)
   - Or manually via `/retrain` endpoint

4. **Retraining Phase** (Minutes 5-30)
   - Fetch all labeled data from database
   - Split: 70% train, 10% val, 20% test
   - Retrain XGBoost model
   - Evaluate on test set
   - Create new model version

5. **Deployment Decision** (Automatic)
   - Check if test accuracy ≥ 80%
   - Check if accuracy drop ≤ 3%
   - If yes → Deploy new model as ACTIVE
   - All new predictions use improved model

### Configuration

Edit `./config/pipeline_config.json`:

```json
{
  "retraining": {
    "enabled": true,
    "threshold_samples": 50,
    "schedule": "daily",
    "time": "02:00"
  },
  "deployment": {
    "auto_deploy": true,
    "min_test_accuracy": 0.80,
    "max_performance_drop": 0.03
  }
}
```

## 🛠️ CLI Commands

### Training Management

```bash
# Manually trigger retraining
python orchestrator.py retrain

# Monitor data distribution
python orchestrator.py monitor

# Generate comprehensive report
python orchestrator.py report

# View current configuration
python orchestrator.py config --show

# Reset configuration to defaults
python orchestrator.py config --reset

# Start continuous scheduler
python orchestrator.py scheduler
```

## 📊 API Endpoints

### Prediction
```
POST /predict
├─ images: 4-5 image files
├─ model_age_months: int
├─ battery_health_pct: float (0-100)
├─ screen_cracked: bool
├─ functional_issues: bool
└─ cosmetic_scratches: int (0-10)

Response:
├─ prediction_id: int (for later feedback)
├─ prediction: str (Reuse/Refurbish/Repair/Recycle)
├─ confidence_pct: float
├─ fused_probability_breakdown: probabilities
├─ individual_votes: image & tabular votes
├─ decision_path: fusion method used
└─ model_version: which model made prediction
```

### Feedback
```
POST /feedback
├─ prediction_id: int
├─ ground_truth: str (Reuse/Refurbish/Repair/Recycle)
└─ user_confidence: float (0-1, optional)
```

### Monitoring
```
GET /model/status          - Current model performance
GET /statistics            - Data & class distribution
GET /predictions/pending-feedback  - Samples waiting for labels
POST /retrain              - Trigger manual retraining
GET /health                - Health check
```

## 🐳 Docker Deployment

### Docker Compose (Local Development)

```bash
# Build and run
docker-compose -f deployment/docker/docker-compose.yml up

# This starts:
# - echoloop-api on :8000
# - echoloop-orchestrator (background scheduler)
```

### Docker Build

```bash
# Build image
docker build -t echoloop:latest -f deployment/docker/Dockerfile .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  echoloop:latest
```

## ☸️ Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f deployment/k8s/deployment.yaml

# Check status
kubectl get deployments
kubectl get pods
kubectl logs -f deployment/echoloop-api

# Scale replicas
kubectl scale deployment echoloop-api --replicas=5
```

## 📈 Monitoring & Observability

### Key Metrics to Monitor

- **Prediction Rate**: Predictions per day/hour
- **Feedback Rate**: % of predictions labeled
- **Model Accuracy**: Train/Val/Test metrics
- **Data Balance**: Class distribution ratio
- **Retraining Frequency**: How often models are updated
- **Inference Latency**: P50, P95, P99

### Check Dashboard

```bash
# View system report
python orchestrator.py report

# Output shows:
# ├─ Active Model & Accuracy
# ├─ Pending Feedback Count
# ├─ Retraining Needed: Yes/No
# ├─ Class Distribution (7d & 30d)
# └─ Next Scheduled Retrain Time
```

## 📂 Project Structure

```
.
├── app_updated.py              # FastAPI application (main API)
├── database.py                 # SQLite data store (RAG backend)
├── continuous_training.py      # Model retraining pipeline
├── orchestrator.py             # Pipeline scheduling & monitoring
├── client.py                   # Python client library
├── model.py                    # EfficientNet model definition
├── dataset.py                  # Data loading & transforms
│
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   └── nginx.conf
│   ├── k8s/
│   │   └── deployment.yaml
│   └── monitoring/
│       ├── prometheus.yml
│       └── filebeat.yml
│
├── data/
│   ├── echoloop_data.db        # SQLite database (created at runtime)
│   ├── train/                  # Training data directories
│   └── val/                    # Validation data directories
│
├── checkpoints/                # Model checkpoints
├── models/                     # Trained model versions
├── logs/                       # Application logs
├── config/                     # Pipeline configuration
│
├── PIPELINE_GUIDE.md           # Detailed architecture guide
├── requirements_updated.txt    # Python dependencies
├── setup.sh                    # Setup & deployment script
└── README.md                   # This file
```

## 🔍 Database Schema

### Key Tables

**predictions_log**: Every prediction
- Input features (images, metadata)
- Model outputs (predictions, probabilities)
- User feedback (ground truth labels)

**model_metadata**: Model versions
- Version ID, creation date
- Training data size
- Accuracy metrics

**training_jobs**: Retraining history
- Job ID, status, timestamps
- Training metrics

## 📚 Best Practices

### 1. Data Collection
- Aim for ≥50 labeled samples before retraining
- Keep classes balanced (max 3:1 ratio)
- Target >50% feedback rate

### 2. Model Deployment
- Set reasonable accuracy thresholds (≥80%)
- Test before production deployment
- Monitor performance post-deployment

### 3. Monitoring
- Check class distribution weekly
- Alert on data imbalance
- Track feedback rate
- Monitor inference latency

### 4. Maintenance
- Archive old predictions periodically
- Version all model updates
- Keep deployment configs in sync
- Regular database backups

## 🚨 Troubleshooting

### API won't start
```bash
# Check port availability
lsof -i :8000

# Check Python version
python --version

# Verify dependencies
pip list | grep torch
```

### Retrain not triggering
```bash
# Check if enough labels collected
curl http://localhost:8000/predictions/pending-feedback

# Manually trigger
python orchestrator.py retrain
```

### Database errors
```bash
# Check database file
ls -lah ./data/echoloop_data.db

# Re-initialize
rm ./data/echoloop_data.db
python -c "from database import ECholooopDataStore; ECholooopDataStore()"
```

## 📞 Support & Contribution

For issues, suggestions, or contributions, please:
1. Check the [PIPELINE_GUIDE.md](./PIPELINE_GUIDE.md) for detailed documentation
2. Run `python orchestrator.py report` for system status
3. Check logs: `tail -f ./logs/orchestrator.log`

## 📄 License

[Your License Here]

## 🎯 Next Steps

1. **Train Initial Models**
   ```bash
   python train_xgboost.py
   python train.ipynb
   ```

2. **Deploy System**
   ```bash
   bash setup.sh setup
   bash setup.sh api  # Terminal 1
   bash setup.sh orchestrator  # Terminal 2
   ```

3. **Start Collecting Data**
   - Use the Python client to make predictions
   - Submit feedback to improve the model
   - Monitor progress via `/statistics` endpoint

4. **Scale to Production**
   ```bash
   bash setup.sh docker  # or
   bash setup.sh k8s
   ```

## 📖 Additional Resources

- [System Architecture Guide](./PIPELINE_GUIDE.md)
- [API Documentation](#-api-endpoints)
- [Deployment Configuration](./deployment_config.py)
- [Python Client Library](./client.py)

---

**Built with ❤️ for continuous machine learning excellence**
