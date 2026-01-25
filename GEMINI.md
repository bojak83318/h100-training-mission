# SysArch H100 Training Mission Context

## Project Overview
This workspace is designated for the **GLM-4.7 Flash Rust Agent (Hybrid NVFP4/FP16)** training mission. The objective is to train a model capable of Rust tool calling using a specific hybrid precision configuration on an NVIDIA H100 GPU.

**Mission Code:** BMAD-2026-Q1
**Authorization:** TPM-BMAD-2026-Q1-GO

## Key Artifacts

### 1. `Execution.md` (Master Artifact Pack)
This file is the single source of truth for the deployment. It contains the exact content for:
*   **`launch_training.sh`**: The orchestrator script for data generation, validation, and training launch.
*   **`config/train_glm4_rust_v3_FINAL.yaml`**: The LLaMA-Factory configuration file, critical for defining the hybrid precision and protecting specific modules (Router, Attention) from quantization.
*   **`scripts/01_transpile_glaive_to_glm4.py`**: A script to convert the Glaive dataset into the GLM-4 native `[TOOL_CALLS]` format.
*   **`scripts/02_monitor_training.py`**: A safety monitor that kills training if loss explodes, stalls, or VRAM usage spikes.
*   **`monitor_heartbeat.sh`**: A utility for real-time log visualization.

### 2. `README.md`
Contains high-level mission status and confirmation of the artifact pack validation.

### 3. `proxmox_mcp.log`
Log file related to Proxmox MCP interactions (likely infrastructure/setup context).

## intended Directory Structure (To Be Created)
The `Execution.md` file prescribes the creation of the following structure:

```text
/home/rocm/workspace/h100/
├── config/
│   └── train_glm4_rust_v3_FINAL.yaml
├── scripts/
│   ├── 01_transpile_glaive_to_glm4.py
│   └── 02_monitor_training.py
├── data/
│   └── (Generated during runtime: rust_agent_native_format.jsonl, etc.)
├── outputs/
│   └── (Training logs and checkpoints)
├── launch_training.sh
└── monitor_heartbeat.sh
```

## Operational Procedures

1.  **Deployment:** The immediate task is often to materialize the files defined in `Execution.md` if they do not exist.
2.  **Execution:** Run `bash launch_training.sh`.
3.  **Monitoring:**
    *   **Loss Trajectory:** Watch for specific milestones (Step 50 cliff < 4.0, Step 200 < 3.0).
    *   **Resources:** Ensure VRAM < 78GB and throughput > 7k tokens/sec.
    *   **Safety:** The `monitor_training.py` script acts as a kill switch.

## Tools & Environment
*   **Hardware:** NVIDIA H100 (80GB VRAM)
*   **Software:** LLaMA-Factory, Python 3, PyTorch (CUDA), Unsloth (implied by "Safe Mode").
*   **Protocol:** GLM-4 Native Tool Calling (`[TOOL_CALLS]`).
