"""
Deployment and Infrastructure Configuration
Includes Docker setup, environment management, and deployment scripts.
"""

# This file contains deployment configurations and instructions


# ============================================================================
# DOCKER CONFIGURATION - Dockerfile
# ============================================================================

DOCKERFILE = """
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p ./checkpoints ./models ./logs ./data ./config

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "app_updated:app", "--host", "0.0.0.0", "--port", "8000"]
"""


# ============================================================================
# DOCKER COMPOSE - For Local Development
# ============================================================================

DOCKER_COMPOSE = """
version: '3.8'

services:
  echoloop-api:
    build: .
    container_name: echoloop-api
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
      - ./models:/app/models
      - ./logs:/app/logs
      - ./config:/app/config
    environment:
      - TORCH_HOME=/app/models
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  orchestrator:
    build: .
    container_name: echoloop-orchestrator
    volumes:
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
      - ./models:/app/models
      - ./logs:/app/logs
      - ./config:/app/config
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    command: python orchestrator.py scheduler

volumes:
  data:
  models:
  checkpoints:
"""


# ============================================================================
# KUBERNETES DEPLOYMENT MANIFESTS
# ============================================================================

K8S_DEPLOYMENT = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: echoloop-api
  namespace: default
spec:
  replicas: 3
  selector:
    matchLabels:
      app: echoloop-api
  template:
    metadata:
      labels:
        app: echoloop-api
    spec:
      containers:
      - name: api
        image: echoloop:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        volumeMounts:
        - name: data
          mountPath: /app/data
        - name: models
          mountPath: /app/models
        - name: config
          mountPath: /app/config
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: echoloop-data-pvc
      - name: models
        persistentVolumeClaim:
          claimName: echoloop-models-pvc
      - name: config
        configMap:
          name: echoloop-config

---
apiVersion: v1
kind: Service
metadata:
  name: echoloop-api-service
  namespace: default
spec:
  selector:
    app: echoloop-api
  type: LoadBalancer
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000

---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: echoloop-retrain
  namespace: default
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: retrain
            image: echoloop:latest
            command: ["python", "orchestrator.py", "retrain"]
            volumeMounts:
            - name: data
              mountPath: /app/data
            - name: models
              mountPath: /app/models
          volumes:
          - name: data
            persistentVolumeClaim:
              claimName: echoloop-data-pvc
          - name: models
            persistentVolumeClaim:
              claimName: echoloop-models-pvc
          restartPolicy: OnFailure

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: echoloop-data-pvc
  namespace: default
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: echoloop-models-pvc
  namespace: default
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
"""


# ============================================================================
# Environment Configuration
# ============================================================================

ENV_EXAMPLE = """
# Echoloop Environment Configuration

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=False

# Model Configuration
MODEL_DEVICE=cuda
MODEL_CHECKPOINT_PATH=./checkpoints/best_model.pth
XGBOOST_MODEL_PATH=./xgboost_model.json

# Database Configuration
DATABASE_PATH=./data/echoloop_data.db

# Retraining Configuration
RETRAINING_THRESHOLD=50
AUTO_RETRAIN_ENABLED=True
RETRAIN_SCHEDULE_TIME=02:00

# Monitoring
MONITORING_ENABLED=True
MONITORING_INTERVAL_HOURS=6
ALERT_THRESHOLD=0.05

# Data Retention
DATA_RETENTION_DAYS=365
"""


# ============================================================================
# NGINX Configuration for Production
# ============================================================================

NGINX_CONFIG = """
upstream echoloop_api {
    server echoloop-api:8000;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/m;
limit_req_zone $binary_remote_addr zone=predict_limit:10m rate=1000r/m;

server {
    listen 80;
    server_name _;

    client_max_body_size 100M;

    # Logging
    access_log /var/log/nginx/echoloop_access.log;
    error_log /var/log/nginx/echoloop_error.log;

    # Health check endpoint - no rate limit
    location /health {
        proxy_pass http://echoloop_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        access_log off;
    }

    # Prediction endpoint - high rate limit
    location /predict {
        limit_req zone=predict_limit burst=200 nodelay;
        proxy_pass http://echoloop_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
        proxy_connect_timeout 60s;
    }

    # Feedback endpoint - moderate rate limit
    location /feedback {
        limit_req zone=api_limit burst=50 nodelay;
        proxy_pass http://echoloop_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # All other endpoints - moderate rate limit
    location / {
        limit_req zone=api_limit burst=50 nodelay;
        proxy_pass http://echoloop_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Static files cache
    location ~* \\.(json|txt|log)$ {
        expires 1h;
    }
}
"""


# ============================================================================
# Monitoring Dashboard Configuration (Prometheus + Grafana)
# ============================================================================

PROMETHEUS_CONFIG = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'echoloop-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
"""


# ============================================================================
# Logging Configuration (ELK Stack)
# ============================================================================

FILEBEAT_CONFIG = """
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /app/logs/*.log

output.elasticsearch:
  hosts: ["elasticsearch:9200"]

processors:
  - add_kubernetes_metadata: ~
  - add_docker_metadata: ~
"""


if __name__ == "__main__":
    """Export configurations to files for deployment."""
    import os
    
    # Create deployment directory
    os.makedirs("./deployment", exist_ok=True)
    os.makedirs("./deployment/k8s", exist_ok=True)
    os.makedirs("./deployment/docker", exist_ok=True)
    os.makedirs("./deployment/monitoring", exist_ok=True)
    
    # Export Dockerfile
    with open("./deployment/docker/Dockerfile", "w") as f:
        f.write(DOCKERFILE)
    
    # Export Docker Compose
    with open("./deployment/docker/docker-compose.yml", "w") as f:
        f.write(DOCKER_COMPOSE)
    
    # Export Kubernetes manifests
    with open("./deployment/k8s/deployment.yaml", "w") as f:
        f.write(K8S_DEPLOYMENT)
    
    # Export NGINX config
    with open("./deployment/docker/nginx.conf", "w") as f:
        f.write(NGINX_CONFIG)
    
    # Export environment example
    with open(".env.example", "w") as f:
        f.write(ENV_EXAMPLE)
    
    # Export monitoring configs
    with open("./deployment/monitoring/prometheus.yml", "w") as f:
        f.write(PROMETHEUS_CONFIG)
    
    with open("./deployment/monitoring/filebeat.yml", "w") as f:
        f.write(FILEBEAT_CONFIG)
    
    print("✓ Deployment configurations exported successfully!")
    print("  - Dockerfile: ./deployment/docker/Dockerfile")
    print("  - Docker Compose: ./deployment/docker/docker-compose.yml")
    print("  - Kubernetes: ./deployment/k8s/deployment.yaml")
    print("  - NGINX: ./deployment/docker/nginx.conf")
    print("  - Monitoring: ./deployment/monitoring/")
