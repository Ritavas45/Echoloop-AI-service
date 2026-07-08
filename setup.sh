#!/bin/bash

# ============================================================================
# Echoloop AI - Complete Setup and Deployment Script
# Automates the entire system setup, training, and deployment
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

print_section() {
    echo -e "\n${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}█ $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# ============================================================================
# Setup Functions
# ============================================================================

setup_directories() {
    print_section "Setting up directories"
    
    mkdir -p ./data
    mkdir -p ./checkpoints
    mkdir -p ./models
    mkdir -p ./logs
    mkdir -p ./config
    mkdir -p ./deployment/docker
    mkdir -p ./deployment/k8s
    mkdir -p ./deployment/monitoring
    
    print_success "Directories created"
}

setup_python_env() {
    print_section "Setting up Python environment"
    
    # Check Python version
    python_version=$(python3 --version | cut -d' ' -f2)
    print_success "Python version: $python_version"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_warning "Virtual environment not found. Creating..."
        python3 -m venv venv
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    print_success "pip upgraded"
    
    # Install dependencies
    pip install -r requirements.txt
    print_success "Dependencies installed"
}

init_database() {
    print_section "Initializing database"
    
    PYTHONPATH=src python3 << 'EOF'
from database import ECholooopDataStore
try:
    store = ECholooopDataStore()
    print("✓ Database initialized successfully")
    print(f"  Database path: ./data/echoloop_data.db")
except Exception as e:
    print(f"✗ Database initialization failed: {e}")
    exit(1)
EOF
}

setup_deployment_configs() {
    print_section "Generating deployment configurations"
    
    PYTHONPATH=src python3 scripts/deployment_config.py
    print_success "Deployment configurations generated"
}

# ============================================================================
# Training Functions
# ============================================================================

train_initial_models() {
    print_section "Training initial models"
    
    if [ ! -f "./xgboost_model.json" ] || [ ! -f "./checkpoints/best_model.pth" ]; then
        print_warning "Models not found. Please run training scripts first:"
        echo "  1. python train_xgboost.py"
        echo "  2. python train.ipynb (or Jupyter notebook)"
        return 1
    fi
    
    print_success "Models found and ready"
}

# ============================================================================
# Application Functions
# ============================================================================

test_api() {
    print_section "Testing API"
    
    # Check if API is running
    if ! curl -s http://localhost:8000/health > /dev/null; then
        print_error "API is not running at http://localhost:8000"
        return 1
    fi
    
    print_success "API is healthy"
    
    # Test status endpoint
    curl -s http://localhost:8000/model/status | python3 -m json.tool
}

start_api() {
    print_section "Starting API server"
    
    echo "API will start on http://localhost:8000"
    echo "Press Ctrl+C to stop"
    echo ""
    
    python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
}

start_orchestrator() {
    print_section "Starting Pipeline Orchestrator"
    
    echo "Orchestrator will start continuous scheduling"
    echo "Press Ctrl+C to stop"
    echo ""
    
    python3 orchestrator.py scheduler
}

start_docker() {
    print_section "Starting with Docker Compose"
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose not installed"
        return 1
    fi
    
    docker-compose -f deployment/docker/docker-compose.yml up
}

# ============================================================================
# Deployment Functions
# ============================================================================

deploy_kubernetes() {
    print_section "Deploying to Kubernetes"
    
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl not installed"
        return 1
    fi
    
    # Check if running on Kubernetes
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Not connected to Kubernetes cluster"
        return 1
    fi
    
    print_warning "Applying Kubernetes manifests..."
    kubectl apply -f deployment/k8s/deployment.yaml
    
    print_success "Kubernetes deployment started"
    echo ""
    echo "Check status with:"
    echo "  kubectl get deployments"
    echo "  kubectl get pods"
    echo "  kubectl logs -f deployment/echoloop-api"
}

# ============================================================================
# CLI Menu
# ============================================================================

show_menu() {
    echo ""
    echo -e "${BLUE}Echoloop AI - Setup & Deployment${NC}"
    echo ""
    echo "1. Full Setup (directories + dependencies + database)"
    echo "2. Initialize Database Only"
    echo "3. Test API"
    echo "4. Start API (development)"
    echo "5. Start Orchestrator (background scheduler)"
    echo "6. Start with Docker Compose"
    echo "7. Deploy to Kubernetes"
    echo "8. Generate Deployment Configs"
    echo "9. View Configuration"
    echo "10. View Documentation"
    echo "0. Exit"
    echo ""
    read -p "Select option: " choice
}

view_config() {
    print_section "Current Configuration"
    python3 orchestrator.py config --show
}

view_docs() {
    print_section "Pipeline Documentation"
    python3 -c "from PIPELINE_GUIDE import ARCHITECTURE_DOCS; print(ARCHITECTURE_DOCS)" | less -R
}

# ============================================================================
# Main Program
# ============================================================================

main() {
    if [ $# -gt 0 ]; then
        # Command-line mode
        case "$1" in
            setup)
                setup_directories
                setup_python_env
                init_database
                setup_deployment_configs
                print_success "Setup complete!"
                ;;
            api)
                start_api
                ;;
            orchestrator)
                start_orchestrator
                ;;
            docker)
                start_docker
                ;;
            k8s)
                deploy_kubernetes
                ;;
            test)
                test_api
                ;;
            *)
                echo "Usage: $0 {setup|api|orchestrator|docker|k8s|test}"
                exit 1
                ;;
        esac
    else
        # Interactive menu
        while true; do
            show_menu
            case $choice in
                1)
                    setup_directories
                    setup_python_env
                    init_database
                    setup_deployment_configs
                    print_success "Full setup complete!"
                    ;;
                2)
                    init_database
                    ;;
                3)
                    test_api
                    ;;
                4)
                    start_api
                    ;;
                5)
                    start_orchestrator
                    ;;
                6)
                    start_docker
                    ;;
                7)
                    deploy_kubernetes
                    ;;
                8)
                    setup_deployment_configs
                    ;;
                9)
                    view_config
                    ;;
                10)
                    view_docs
                    ;;
                0)
                    print_success "Goodbye!"
                    exit 0
                    ;;
                *)
                    print_error "Invalid option"
                    ;;
            esac
        done
    fi
}

# ============================================================================
# Run Main
# ============================================================================

main "$@"
