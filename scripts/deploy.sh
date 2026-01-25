#!/bin/bash
# H100 Deployment & Setup Script
# Authorization: TPM-BMAD-2026-Q1-GO

set -e

echo "🚀 Starting SysArch H100 Environment Setup..."

# 1. System Dependencies
echo "📦 Installing system dependencies..."
sudo apt-get update && sudo apt-get install -y git python3 python3-pip python3-venv bc screen

# 2. Python Environment
echo "🐍 Setting up Python environment..."
# Check if we are already in a venv, if not create one
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Activated venv."
fi

# 3. Python Packages
echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Unsloth is hardware specific, try standard install or warn
echo "⚡ Installing Unsloth (this may take a moment)..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" || echo "⚠️ Unsloth install failed or requires manual intervention. Proceeding..."

# 4. Permissions
echo "🔒 Setting permissions..."
chmod +x launch_training.sh
chmod +x monitor_heartbeat.sh
chmod +x scripts/*.py

# 5. LLaMA Factory Check
if ! command -v llamafactory-cli &> /dev/null; then
    echo "⚠️ llamafactory-cli not found. Installing from source..."
    git clone https://github.com/hiyouga/LLaMA-Factory.git || echo "LLaMA-Factory dir exists."
    pip install -e LLaMA-Factory
fi

echo "✅ Environment Setup Complete."
echo "   Run: ./launch_training.sh to start the mission."
