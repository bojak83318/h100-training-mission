# SysArch H100 Training Mission

**Mission Code:** BMAD-2026-Q1
**Status:** 🟢 READY FOR DEPLOYMENT

This repository contains the complete infrastructure code for training the **GLM-4.7 Flash Rust Agent** on an NVIDIA H100 GPU (specifically targeted for OVH Cloud instances).

## 🚀 Quick Start (OVH Cloud H100)

**Step 1: Connect to your Instance**
SSH into your fresh H100 machine.

**Step 2: One-Liner Deployment**
Copy and paste this command block to clone the repo, setup the environment, and prepare for launch.

```bash
# Clone the mission repository
git clone https://github.com/bojak83318/h100-training-mission.git
cd h100-training-mission

# Run the automated setup (installs deps, venv, unsloth, llamafactory)
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

**Step 3: Launch Mission**
Once the setup is complete:

```bash
# Activate venv if not already active
source venv/bin/activate

# Initiate the training pipeline
./launch_training.sh
```

## 📂 Repository Structure

*   `scripts/`: Automation scripts (transpilers, monitors, setup).
*   `config/`: LLaMA-Factory configuration files (Hybrid FP4/FP16 protection).
*   `data/`: Generated datasets (transpiled locally).
*   `outputs/`: Training logs and model checkpoints.
*   `launch_training.sh`: The main orchestrator.
*   `monitor_heartbeat.sh`: Real-time CLI dashboard.

## 🛡️ Safety Systems
The `launch_training.sh` script automatically engages:
1.  **VRAM Protection:** Downgrades batch size if < 75GB VRAM detected.
2.  **Kill Switch (`scripts/02_monitor_training.py`):** Terminates training on loss explosion (>8.0) or cliff misses (Step 50 > 4.0).
3.  **Router Protection:** Specifically skips quantization for `moe.router` and attention layers to prevent MoE collapse.

## 📊 Monitoring
During training, open a new terminal and run:
```bash
./monitor_heartbeat.sh
```
