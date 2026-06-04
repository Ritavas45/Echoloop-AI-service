#!/bin/sh

# ============================================================================
# Echoloop AI - Container Entrypoint Script
# Handles persistent volume setup and migrations on cloud deployments
# ============================================================================

PERSISTENT_DIR="/app/persistent"

if [ -d "$PERSISTENT_DIR" ]; then
    echo "[Entrypoint] Persistent storage detected at $PERSISTENT_DIR"
    
    # Create persistent subdirectories
    mkdir -p "$PERSISTENT_DIR/data"
    mkdir -p "$PERSISTENT_DIR/checkpoints"
    mkdir -p "$PERSISTENT_DIR/models"
    mkdir -p "$PERSISTENT_DIR/logs"
    
    # 1. Migrate Database
    if [ ! -f "$PERSISTENT_DIR/data/echoloop_data.db" ] && [ -f "/app/data/echoloop_data.db" ]; then
        echo "[Entrypoint] Initializing persistent database with local copy..."
        cp "/app/data/echoloop_data.db" "$PERSISTENT_DIR/data/"
    fi
    
    # 2. Migrate Checkpoints
    if [ ! -f "$PERSISTENT_DIR/checkpoints/best_model.pth" ] && [ -f "/app/checkpoints/best_model.pth" ]; then
        echo "[Entrypoint] Initializing persistent checkpoints with local copy..."
        cp "/app/checkpoints/best_model.pth" "$PERSISTENT_DIR/checkpoints/"
    fi
    
    # 3. Swap directories with symlinks
    echo "[Entrypoint] Linking app directories to persistent storage..."
    rm -rf /app/data /app/checkpoints /app/models /app/logs
    ln -s "$PERSISTENT_DIR/data" /app/data
    ln -s "$PERSISTENT_DIR/checkpoints" /app/checkpoints
    ln -s "$PERSISTENT_DIR/models" /app/models
    ln -s "$PERSISTENT_DIR/logs" /app/logs
    
    echo "[Entrypoint] Persistent storage linked successfully."
else
    echo "[Entrypoint] Running in ephemeral mode (No persistent volume detected at $PERSISTENT_DIR)."
fi

# Start the application
echo "[Entrypoint] Launching application..."
exec python app_updated.py
