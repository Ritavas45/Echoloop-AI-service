#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$DIR")"
SDK_DIR="$ROOT_DIR/gcloud_sdk"

echo "=== Setting up Google Cloud SDK in $SDK_DIR ==="

mkdir -p "$SDK_DIR"
cd "$SDK_DIR"

if [ -d "google-cloud-sdk" ]; then
    echo "google-cloud-sdk directory already exists. Skipping download."
else
    echo "Downloading Google Cloud CLI for macOS arm64..."
    curl -o google-cloud-cli.tar.gz https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-darwin-arm.tar.gz
    
    echo "Extracting archive..."
    tar -xzf google-cloud-cli.tar.gz
    rm google-cloud-cli.tar.gz
    
    echo "Running installation script..."
    ./google-cloud-sdk/install.sh --quiet --path-update false --usage-reporting false
fi

echo "Google Cloud CLI set up successfully."
echo "gcloud executable is at: $SDK_DIR/google-cloud-sdk/bin/gcloud"
