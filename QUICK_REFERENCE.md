
# ECHOLOOP AI - QUICK REFERENCE GUIDE

## 🚀 Start System (3 steps)

### Terminal 1: Start API
```bash
cd /path/to/Echoloop-AI-service
source venv/bin/activate
python -m uvicorn app_updated:app --host 0.0.0.0 --port 8000
```

### Terminal 2: Start Orchestrator
```bash
cd /path/to/Echoloop-AI-service
source venv/bin/activate
python orchestrator.py scheduler
```

### Terminal 3: Test
```bash
curl http://localhost:8000/health
python client.py
```

---

## 📊 Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/predict` | POST | Make prediction & log it |
| `/feedback` | POST | Submit ground truth label |
| `/model/status` | GET | Current model performance |
| `/statistics` | GET | Data collection metrics |
| `/retrain` | POST | Manual retraining trigger |
| `/health` | GET | API health check |

---

## 💻 CLI Commands

```bash
# Trigger retraining
python orchestrator.py retrain

# Monitor data distribution
python orchestrator.py monitor

# Generate report
python orchestrator.py report

# View/manage config
python orchestrator.py config --show
python orchestrator.py config --reset

# Start continuous scheduler
python orchestrator.py scheduler
```

---

## 🔄 Continuous Learning Workflow

```
Predictions → Database → Feedback Collection → Automatic Retrain → Deployment
   (Min 1)      (Store)      (50+ labels)        (5-30 min)        (Auto)
```

---

## 📁 Important Directories

```
./data/              # SQLite database & logs
./checkpoints/       # Latest model checkpoint
./models/            # All model versions
./config/            # Pipeline configuration
./logs/              # Application logs
./deployment/        # Docker & K8s configs
```

---

## 🔧 Configuration

Edit `./config/pipeline_config.json`:

```json
{
  "retraining": {
    "threshold_samples": 50,     # Labels to trigger retrain
    "time": "02:00"              # Daily schedule time
  },
  "deployment": {
    "auto_deploy": false,        # Auto-deploy if criteria met
    "min_test_accuracy": 0.80,   # Minimum accuracy
    "max_performance_drop": 0.03 # Max acceptable accuracy drop
  }
}
```

---

## 📈 Monitor System Health

```bash
# All-in-one report
python orchestrator.py report

# Check predictions waiting for feedback
curl http://localhost:8000/predictions/pending-feedback

# View logs
tail -f ./logs/orchestrator.log

# Check database
sqlite3 ./data/echoloop_data.db "SELECT COUNT(*) FROM predictions_log;"
```

---

## 🐳 Docker Quick Start

```bash
# Build
docker build -t echoloop:latest .

# Run
docker-compose up

# Access
curl http://localhost:8000/health
```

---

## ☸️ Kubernetes Quick Deploy

```bash
kubectl apply -f deployment/k8s/deployment.yaml
kubectl get deployments
kubectl logs -f deployment/echoloop-api
```

---

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8000 in use | `lsof -i :8000` and kill process |
| Models not found | Run `train_xgboost.py` or `train.ipynb` |
| DB error | `rm ./data/echoloop_data.db` then restart |
| No retrain | Check if ≥50 labels: `/predictions/pending-feedback` |
| Slow inference | Check GPU: `python -c "import torch; print(torch.cuda.is_available())"` |

---

## 📊 Database Schema Quick View

```sql
-- predictions_log: Every prediction
SELECT COUNT(*) as total_predictions FROM predictions_log;
SELECT COUNT(*) as pending_labels FROM predictions_log WHERE feedback_received = 0;
SELECT ground_truth, COUNT(*) FROM predictions_log WHERE feedback_received = 1 GROUP BY ground_truth;

-- model_metadata: Model versions
SELECT model_version, test_accuracy, created_date FROM model_metadata ORDER BY created_date DESC;

-- training_jobs: Retraining history
SELECT status, COUNT(*) FROM training_jobs GROUP BY status;
```

---

## 🎯 Success Checklist

- [ ] API starts on :8000
- [ ] Orchestrator runs background scheduler
- [ ] Can make predictions via `/predict`
- [ ] Can submit feedback via `/feedback`
- [ ] Model status shows correctly
- [ ] 50+ labels collected → Retraining triggers
- [ ] New model deployed automatically
- [ ] System logs appear in ./logs/

---

## 🔗 Connected Services

| Service | Port | URL | Logs |
|---------|------|-----|------|
| API | 8000 | `http://localhost:8000` | `Terminal 1` |
| Orchestrator | - | - | `./logs/orchestrator.log` |
| Database | - | `./data/echoloop_data.db` | SQLite |

---

## 🚀 Performance Targets

- **Prediction Speed**: <500ms per prediction
- **Retrain Time**: 5-30 minutes
- **Feedback Rate**: >50% of predictions
- **Model Accuracy**: ≥80% on test set
- **Uptime**: >99.9% (excluding maintenance)

---

## 📚 Full Documentation

- [Architecture Guide](./PIPELINE_GUIDE.md)
- [Implementation Guide](./IMPLEMENTATION.md)
- [API Reference](./README_CONTINUOUS_LEARNING.md)
- [Setup Instructions](./setup.sh)

---

## 💡 Pro Tips

1. **Collect balanced data**: Keep class distribution 1:1 to 3:1
2. **User feedback quality**: Encourage high confidence (>0.9)
3. **Model versioning**: Keep old versions for rollback
4. **Monitor regularly**: Check report daily in production
5. **Archive data**: Move old predictions to separate storage monthly

---

**Generated**: $(date)
**System Version**: 2.0
**Last Updated**: June 2024
