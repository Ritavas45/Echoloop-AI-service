
# ECHOLOOP AI - COMPLETE IMPLEMENTATION GUIDE

## Phase 1: Setup & Configuration (1-2 hours)

### Step 1.1: Verify Prerequisites
- [ ] Python 3.10+ installed
- [ ] 8GB+ RAM available
- [ ] 100GB+ disk space for data & models
- [ ] GPU recommended (CUDA/MPS) but not required

### Step 1.2: Clone/Setup Repository
```bash
cd /path/to/Echoloop-AI-service
pwd  # Verify path
```

### Step 1.3: Run Full Setup
```bash
# Make setup script executable
chmod +x setup.sh

# Run interactive setup (choose option 1)
./setup.sh

# Or use command line
./setup.sh setup
```

This will:
- Create ./data, ./checkpoints, ./models, ./logs, ./config
- Create Python virtual environment
- Install all dependencies from requirements_updated.txt
- Initialize SQLite database
- Generate Docker & Kubernetes configs

### Step 1.4: Verify Installation
```bash
# Activate venv
source venv/bin/activate

# Test imports
python -c "import torch; print(f'Torch: {torch.__version__}')"
python -c "import xgboost; print(f'XGBoost: {xgboost.__version__}')"
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"

# Verify database
python -c "from database import ECholooopDataStore; print('✓ Database initialized')"
```

## Phase 2: Model Training (if needed) (2-4 hours)

### Step 2.1: Check if Models Exist
```bash
ls -lh ./checkpoints/best_model.pth
ls -lh ./xgboost_model.json
```

### Step 2.2: Train Models (if missing)

**Option A: Using Existing Notebooks**
```bash
# Image model training (EfficientNet)
jupyter notebook train.ipynb

# Tabular model training (XGBoost)
python train_xgboost.py
```

**Option B: Quick Test Models**
```python
# Create dummy models for testing
import torch
import xgboost as xgb
from model import LateFusionEfficientNet

# Create dummy image model
image_model = LateFusionEfficientNet(num_classes=4)
torch.save({'model_state_dict': image_model.state_dict()}, './checkpoints/best_model.pth')
print('✓ Dummy image model saved')

# Create dummy XGBoost
X_dummy = [[24, 75, 0, 0, 2], [36, 60, 1, 0, 5]]
y_dummy = ['Refurbish', 'Repair']
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_dummy, y_dummy)
xgb_model.save_model('./xgboost_model.json')
print('✓ Dummy XGBoost model saved')
```

### Step 2.3: Verify Models Load
```bash
python -c "
import torch
from model import LateFusionEfficientNet
device = torch.device('cpu')
model = LateFusionEfficientNet(num_classes=4)
checkpoint = torch.load('./checkpoints/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print('✓ Image model loads successfully')
"

python -c "
import xgboost as xgb
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('./xgboost_model.json')
print('✓ XGBoost model loads successfully')
"
```

## Phase 3: Running the System (5-10 minutes)

### Step 3.1: Terminal Setup
You'll need 3 terminals open:
```
Terminal 1: FastAPI Server
Terminal 2: Orchestrator Scheduler
Terminal 3: Testing/Monitoring
```

### Step 3.2: Start FastAPI Server (Terminal 1)
```bash
cd /path/to/Echoloop-AI-service
source venv/bin/activate
python -m uvicorn app_updated:app --host 0.0.0.0 --port 8000 --reload

# Output should show:
# ✓ Uvicorn running on http://0.0.0.0:8000
# ✓ Application startup complete
```

### Step 3.3: Start Orchestrator (Terminal 2)
```bash
cd /path/to/Echoloop-AI-service
source venv/bin/activate
python orchestrator.py scheduler

# Output should show:
# [Orchestrator] Starting Pipeline Scheduler...
# [Orchestrator] Scheduled daily retraining at 02:00
# [Orchestrator] Scheduled data monitoring every 6 hours
# [Orchestrator] Scheduler running...
```

### Step 3.4: Test System (Terminal 3)
```bash
# Health check
curl http://localhost:8000/health

# Get model status
curl http://localhost:8000/model/status | python -m json.tool

# View stats
curl http://localhost:8000/statistics | python -m json.tool
```

## Phase 4: Making First Predictions (10-15 minutes)

### Step 4.1: Prepare Test Data
```bash
# Create test images (or use existing)
mkdir -p ./test_data
# Add 4 test phone images: phone_test_1.jpg, phone_test_2.jpg, etc.
```

### Step 4.2: Make a Prediction via cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "images=@test_data/phone_test_1.jpg" \
  -F "images=@test_data/phone_test_2.jpg" \
  -F "images=@test_data/phone_test_3.jpg" \
  -F "images=@test_data/phone_test_4.jpg" \
  -F "model_age_months=24" \
  -F "battery_health_pct=75" \
  -F "screen_cracked=false" \
  -F "functional_issues=false" \
  -F "cosmetic_scratches=2"
```

Response:
```json
{
  "prediction_id": 1,
  "prediction": "Refurbish",
  "confidence_pct": 82.5,
  "fused_probability_breakdown": {
    "Reuse": 0.05,
    "Refurbish": 0.825,
    "Repair": 0.10,
    "Recycle": 0.025
  },
  "individual_votes": {
    "image_model_prediction": "Refurbish",
    "image_model_confidence_pct": 80.0,
    "tabular_model_prediction": "Refurbish",
    "tabular_model_confidence_pct": 85.0
  },
  "decision_path": "weighted_soft_voting",
  "model_version": "v1_base",
  "timestamp": "2024-06-01T10:30:45.123456"
}
```

### Step 4.3: Submit Feedback
```bash
curl -X POST "http://localhost:8000/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_id": 1,
    "ground_truth": "Refurbish",
    "user_confidence": 0.95
  }'

# Response:
# {"success": true, "message": "Feedback recorded...", "prediction_id": 1}
```

### Step 4.4: Using Python Client
```python
from client import EcholooopClient

# Initialize client
client = EcholooopClient("http://localhost:8000")

# Make prediction
result = client.predict(
    image_paths=[
        "test_data/phone_test_1.jpg",
        "test_data/phone_test_2.jpg",
        "test_data/phone_test_3.jpg",
        "test_data/phone_test_4.jpg"
    ],
    model_age_months=24,
    battery_health_pct=75.0,
    screen_cracked=False,
    functional_issues=False,
    cosmetic_scratches=2
)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence_pct']:.1f}%")
pred_id = result['prediction_id']

# Submit feedback
feedback_result = client.submit_feedback(
    prediction_id=pred_id,
    ground_truth="Refurbish",
    user_confidence=0.95
)
print("Feedback submitted!")

# Check model status
status = client.get_model_status()
print(f"Model: {status['active_model_version']}")
print(f"Accuracy: {status['accuracy_test']:.2%}")
```

## Phase 5: Continuous Learning Loop (Ongoing)

### Step 5.1: Collect Predictions
```
Days 1-4:
- Users make predictions (20-30 per day)
- System logs each prediction
- Each gets a unique prediction_id
```

### Step 5.2: Collect Feedback
```
Days 1-4:
- Users verify predictions
- Submit correct labels via /feedback endpoint
- System counts labeled predictions
```

### Step 5.3: Monitor Progress
```bash
# Terminal 3: Check pending feedback count
curl http://localhost:8000/predictions/pending-feedback | python -m json.tool

# Example output: 45 pending predictions
# If you collect 50+ labels → Retraining will trigger!
```

### Step 5.4: Automatic Retraining Trigger
```
DAY 5 02:00 AM:
- Orchestrator sees 50+ new labels
- Triggers retraining pipeline
- Trains XGBoost on collected data
- Evaluates on test set
- If criteria met → Deploys new model
- All new predictions use improved model
```

### Step 5.5: Monitor Retraining
```bash
# Check if retraining is needed
curl http://localhost:8000/model/status | python -m json.tool

# Output shows:
# "retraining_required": false  # New model deployed!

# Check logs
tail -f ./logs/orchestrator.log

# View training report
python orchestrator.py report
```

## Phase 6: Production Deployment (Optional)

We unified the FastAPI server and the orchestrator scheduler into a single process. Therefore, you only need to run a single container instance in production, simplifying deployment and avoiding shared volume conflicts. Below are the recommended deployment paths:

### Path A: Render Deployment (Fastest & Easiest)

Render provides an easy, Blueprint-based setup using [render.yaml](file:///Users/ritavas/Desktop/Frontend/Echoloop-AI-service/render.yaml).

1. **Push Code to Git**: Commit and push the project files to a private or public GitHub/GitLab repository.
2. **Connect to Render**:
   - Log in to [render.com](https://render.com).
   - Go to **Blueprints** -> **New Blueprint Instance**.
   - Connect your repository. Render will read the [render.yaml](file:///Users/ritavas/Desktop/Frontend/Echoloop-AI-service/render.yaml) file.
3. **Automatic Provisioning**:
   - Render will create a Web Service called `echoloop-ai-service`.
   - It will attach a 10GB persistent disk at `/app/persistent` (which our [entrypoint.sh](file:///Users/ritavas/Desktop/Frontend/Echoloop-AI-service/deployment/docker/entrypoint.sh) script automatically links to persist the SQLite database, model checkpoints, and logs).
   - Render will build the container using [Dockerfile](file:///Users/ritavas/Desktop/Frontend/Echoloop-AI-service/deployment/docker/Dockerfile) and deploy it.
4. **Access Endpoint**: Your service will be online at `https://echoloop-ai-service.onrender.com/health`.

---

### Path B: AWS EC2 or Lightsail Deployment (Recommended for AWS with SQLite)

Deploying on a virtual machine is the most cost-effective way to host stateful containerized ML applications on AWS since it uses a persistent local EBS SSD.

1. **Launch Instance**:
   - Create a `t3.medium` (4GB RAM) EC2 or Lightsail instance running **Ubuntu 22.04 LTS**.
   - Configure the **Security Group** to allow inbound TCP traffic on port `8000` (API) and port `22` (SSH).
2. **Install Docker & Docker Compose**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker.io docker-compose
   sudo systemctl start docker
   sudo systemctl enable docker
   sudo usermod -aG docker $USER  # Log out and log back in for changes to take effect
   ```
3. **Deploy the Code**:
   - Clone the repository onto the instance.
   - Run the service containerized in the background:
     ```bash
     docker-compose -f deployment/docker/docker-compose.yml up --build -d
     ```
4. **Verify**:
   - Check container logs: `docker logs -f echoloop-api`
   - Test health check: `curl http://<INSTANCE_PUBLIC_IP>:8000/health`

---

### Path C: AWS ECS (Elastic Container Service) on Fargate (Enterprise Scale)

For a serverless container setup on AWS, we use ECS Fargate with AWS EFS (Elastic File System) to persist our SQLite database.

1. **Create EFS Volume**:
   - In AWS Console, go to **EFS** -> **Create File System**.
   - Note the File System ID (e.g., `fs-12345678`).
   - Configure EFS Mount Targets in your VPC subnets and allow inbound NFS traffic (port 2049) from your ECS security group.
2. **Push Image to AWS ECR**:
   - Create a repository in **ECR** (Elastic Container Registry) called `echoloop`.
   - Build, tag, and push the image:
     ```bash
     aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
     docker build -t echoloop:latest -f deployment/docker/Dockerfile .
     docker tag echoloop:latest <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/echoloop:latest
     docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/echoloop:latest
     ```
3. **Register ECS Task Definition**:
   Create a Task Definition in ECS with:
   - **Launch Type**: Fargate (1 vCPU, 4GB RAM)
   - **Storage / Volume**: Add a volume named `echoloop-storage` with volume type `EFS` and specify your EFS File System ID.
   - **Container Definition**:
     - Image: `<AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/echoloop:latest`
     - Port Mappings: Host `8000` -> Container `8000`
     - Mount Points: Container path `/app/persistent` mapped to volume `echoloop-storage`
     - Env variables: `PORT=8000`, `PYTHONUNBUFFERED=1`, `TORCH_HOME=/app/models`
4. **Deploy Service**:
   - Create an ECS Service using the Task Definition.
   - Attach an **Application Load Balancer (ALB)** to route public HTTPS traffic to the ECS task on port `8000`.

---

### Path D: Kubernetes Deployment

For Kubernetes deployments (using the manifests in [deployment/k8s/](file:///Users/ritavas/Desktop/Frontend/Echoloop-AI-service/deployment/k8s/)), ensure a PersistentVolumeClaim (PVC) is configured and mounted at `/app/persistent` to allow the SQLite database to survive pod restarts.

```bash
# Apply Kubernetes manifests
kubectl apply -f deployment/k8s/deployment.yaml

# Check deployment
kubectl get deployments
kubectl get pods
kubectl logs -f deployment/echoloop-api
```

## Phase 7: Monitoring & Maintenance (Ongoing)

### Daily Tasks
```bash
# 1. Check system health
curl http://localhost:8000/health

# 2. Review feedback rate
python orchestrator.py monitor

# 3. Check data distribution
python orchestrator.py report
```

### Weekly Tasks
```bash
# 1. Review model performance
curl http://localhost:8000/model/status

# 2. Check class distribution
curl http://localhost:8000/statistics

# 3. Verify predictions are being logged
sqlite3 ./data/echoloop_data.db "SELECT COUNT(*) FROM predictions_log;"
```

### Monthly Tasks
```bash
# 1. Generate comprehensive report
python orchestrator.py report

# 2. Archive old predictions (if needed)
# 3. Update pipeline configuration
# 4. Review model accuracy trends
# 5. Plan data collection strategy
```

## Key Files & Where to Edit

### API Configuration
File: `app_updated.py`
- Lines 27-32: Model paths and configuration
- Line 37: RETRAINING_THRESHOLD (default: 50)
- Line 38: CURRENT_MODEL_VERSION

### Retraining Configuration
File: `./config/pipeline_config.json`
```json
{
  "retraining": {
    "threshold_samples": 50,     # Trigger after 50 labels
    "time": "02:00"              # Daily at 2 AM
  },
  "deployment": {
    "min_test_accuracy": 0.80,   # Don't deploy if < 80%
    "max_performance_drop": 0.03  # Don't deploy if drops > 3%
  }
}
```

### Database Location
- Main database: `./data/echoloop_data.db`
- Models: `./models/xgboost_*.json`
- Checkpoints: `./checkpoints/best_model.pth`
- Logs: `./logs/orchestrator.log`

## Troubleshooting Checklist

### API Won't Start
- [ ] Check Python version: `python --version` (should be 3.10+)
- [ ] Verify port 8000 is free: `lsof -i :8000`
- [ ] Check dependencies: `pip list | grep fastapi`
- [ ] Check GPU: `python -c "import torch; print(torch.cuda.is_available())"`

### Predictions Fail
- [ ] Check image files exist: `ls -lh test_data/*.jpg`
- [ ] Verify models load: `python -c "from model import ..."`
- [ ] Check logs: `tail -f ./logs/orchestrator.log`
- [ ] Test API health: `curl http://localhost:8000/health`

### Retraining Not Triggering
- [ ] Check feedback count: `curl http://localhost:8000/predictions/pending-feedback`
- [ ] Check threshold: View `./config/pipeline_config.json`
- [ ] Manually trigger: `python orchestrator.py retrain`
- [ ] Check logs: `tail -f ./logs/orchestrator.log`

### Database Issues
- [ ] Check file exists: `ls -lh ./data/echoloop_data.db`
- [ ] Check disk space: `df -h ./data/`
- [ ] Verify permissions: `ls -la ./data/`
- [ ] Re-initialize if corrupted: `rm ./data/echoloop_data.db && python -c "from database import ECholooopDataStore; ECholooopDataStore()"`

## Performance Optimization Tips

1. **Model Loading Time**: Keep model checkpoints small
   - Quantize models for production
   - Use model compression techniques

2. **Prediction Speed**: Batch predictions when possible
   - Client should queue multiple predictions
   - API can batch process them

3. **Database**: Index frequently queried columns
   - Add indexes on `model_version`, `feedback_received`
   - Archive old predictions to separate storage

4. **Retraining**: Parallelize data loading
   - Use multiple workers in data loading
   - Consider distributed training for large datasets

## Success Metrics

Track these to ensure system is working well:

1. **Data Collection**
   - [ ] ≥20 predictions per day
   - [ ] ≥50% feedback rate
   - [ ] Balanced class distribution (≤3:1)

2. **Model Performance**
   - [ ] Test accuracy ≥80%
   - [ ] Accuracy improvement after retrain
   - [ ] No accuracy drops >3%

3. **System Health**
   - [ ] API response time <500ms
   - [ ] <1% failed predictions
   - [ ] Successful retraining every 5-7 days

4. **Feedback Quality**
   - [ ] User confidence >0.9
   - [ ] Prediction correctness >90%
   - [ ] No data imbalance issues

## Next Steps After Setup

1. **Integrate with Frontend**
   - Use Python client library (client.py)
   - Build UI for feedback submission
   - Display model confidence to users

2. **Scale Up**
   - Deploy to cloud (AWS/GCP/Azure)
   - Configure load balancing
   - Set up monitoring & alerts

3. **Optimize**
   - Analyze prediction errors
   - Improve feature engineering
   - Fine-tune model hyperparameters

4. **Extend**
   - Add more model architectures
   - Implement active learning
   - Add data augmentation

---

**Estimated Total Time**: 
- Setup: 1-2 hours
- First predictions: 15-30 minutes
- Full cycle (prediction → feedback → retrain): 3-5 days
