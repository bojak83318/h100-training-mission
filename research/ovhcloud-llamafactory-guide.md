# OVHcloud AI Training - LLaMA-Factory Setup Guide

> Research Date: 2026-01-26
> Source: Perplexity AI Deep Research

## **1. Recommended Docker Images (2024-2025 Compatible)**

### **Option A: NVIDIA PyTorch NGC (Recommended for H100)**

**Latest Stable Images:**
```bash
# PyTorch 2.5 + CUDA 12.6 (Latest as of Jan 2025)
nvcr.io/nvidia/pytorch:24.12-py3

# PyTorch 2.4 + CUDA 12.4 (Stable alternative)
nvcr.io/nvidia/pytorch:24.08-py3

# PyTorch 2.3 + CUDA 12.1 (Widely tested)
nvcr.io/nvidia/pytorch:24.01-py3
```

**What's Included:**
- PyTorch 2.x pre-installed
- CUDA 12.x + cuDNN 9.x
- Python 3.10+
- NCCL, TensorRT, DALI
- **Transformers NOT included** (must install via Dockerfile)

**Compatibility:** ✅ Linux/AMD64, works with OVHcloud (requires user 42420 permissions)

---

### **Option B: Hugging Face Official (Best for LLaMA-Factory)**

```bash
# Transformers + PyTorch GPU (Public)
huggingface/transformers-pytorch-gpu:latest

# Specific version (recommended for production)
huggingface/transformers-pytorch-gpu:4.36.0
```

**What's Included:**
- Transformers library pre-installed
- PyTorch with GPU support
- Basic CUDA runtime
- **Missing:** PEFT, bitsandbytes, Unsloth (must add)

---

### **Option C: OVHcloud Preset Images**

OVHcloud provides **pre-configured notebook images** with Transformers, but these are optimized for interactive use (JupyterLab), not batch training jobs.

**Available Presets:**
- `Hugging Face Transformers` (notebook environment)
- `PyTorch` (with JupyterLab + VSCode)

**Limitation:** These are **NOT recommended for production training jobs** as they include unnecessary IDE overhead. Better to build custom image.

---

## **2. Custom Dockerfile for LLaMA-Factory on OVHcloud**

Here's a production-ready Dockerfile that satisfies all OVHcloud requirements:

```dockerfile
# RECOMMENDED: Start from NVIDIA PyTorch NGC (H100-optimized)
FROM nvcr.io/nvidia/pytorch:24.08-py3

# CRITICAL: Set /workspace as HOME (OVHcloud requirement)
WORKDIR /workspace
ENV HOME=/workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install LLaMA-Factory dependencies
RUN pip install --no-cache-dir \
    transformers==4.36.0 \
    peft==0.7.1 \
    bitsandbytes==0.41.3 \
    accelerate==0.25.0 \
    datasets==2.16.0 \
    sentencepiece \
    protobuf \
    tiktoken \
    wandb

# Clone LLaMA-Factory (or copy from local)
RUN git clone https://github.com/hiyouga/LLaMA-Factory.git /workspace/LLaMA-Factory && \
    cd /workspace/LLaMA-Factory && \
    pip install -e .

# CRITICAL: Give OVHcloud user (42420:42420) full access
RUN chown -R 42420:42420 /workspace

# Optional: Copy your training configs
# COPY config/ /workspace/config/

# Optional: Set default command (can override at job launch)
# CMD ["llamafactory-cli", "train", "config/train_glm4_rust_v3_FINAL.yaml"]
```

**Build Command (MUST use linux/amd64):**
```bash
# Build for OVHcloud compatibility
docker buildx build --platform linux/amd64 -t llamafactory-h100:v1.0 .

# Test locally as OVHcloud user
docker run --rm -it --user=42420:42420 --gpus all llamafactory-h100:v1.0 bash
```

---

## **3. Pushing to OVHcloud-Compatible Registry**

### **Option 1: OVHcloud Shared Registry (Quick Testing)**

```bash
# Get credentials from OVHcloud Control Panel > AI Training > Shared Registry
docker login -u <project-id> -p <password> registry.gra.ai.cloud.ovh.net

# Tag and push
docker tag llamafactory-h100:v1.0 registry.gra.ai.cloud.ovh.net/llamafactory-h100:v1.0
docker push registry.gra.ai.cloud.ovh.net/llamafactory-h100:v1.0
```

**⚠️ Warning:** Shared registry is for testing only, not production.

### **Option 2: Docker Hub Private Registry**

```bash
# Add Docker Hub to OVHcloud (via CLI or Control Panel)
ovhai registry add index.docker.io

# Login and push
docker login
docker tag llamafactory-h100:v1.0 <your-dockerhub-username>/llamafactory-h100:v1.0
docker push <your-dockerhub-username>/llamafactory-h100:v1.0
```

**To use in AI Training job:**
```bash
ovhai job run index.docker.io/<username>/llamafactory-h100:v1.0 \
  --gpu 1 \
  --volume <object-storage-path>@GRA:/workspace/data:RW
```

---

## **4. Launching LLaMA-Factory Job on OVHcloud H100**

### **Via CLI (Recommended)**

```bash
# Install OVHcloud AI CLI
pip install ovhai

# Configure credentials (get from Control Panel)
ovhai login

# Launch training job with H100
ovhai job run registry.gra.ai.cloud.ovh.net/llamafactory-h100:v1.0 \
  --name "glm4-rust-training" \
  --gpu 1 \
  --gpu-model H100 \
  --volume <your-bucket>@GRA:/workspace/data:RW \
  --volume <your-bucket>@GRA:/workspace/outputs:RW \
  -- bash -c 'llamafactory-cli train /workspace/config/train_glm4_rust_v3_FINAL.yaml'
```

### **Mount Git Repository (Alternative to Docker COPY)**

```bash
# Clone repo at runtime instead of baking into image
ovhai job run nvcr.io/nvidia/pytorch:24.08-py3 \
  --name "glm4-rust-training" \
  --gpu 1 \
  --gpu-model H100 \
  --volume <data-bucket>@GRA:/workspace/data:RW \
  -- bash -c '
    git clone https://github.com/hiyouga/LLaMA-Factory.git && \
    cd LLaMA-Factory && \
    pip install -e . && \
    llamafactory-cli train /workspace/data/train_glm4_rust_v3_FINAL.yaml
  '
```

---

## **5. Object Storage Integration**

**Link S3-compatible Object Storage to job:**

```bash
# Create Object Storage container first (via Control Panel)
# Example container name: ai-training-data

# Mount at job launch
ovhai job run <your-image> \
  --volume ai-training-data@GRA:/workspace/data:RW \
  --volume ai-training-checkpoints@GRA:/workspace/outputs:RW
```

**Inside job, access data at:**
- `/workspace/data/` (input dataset)
- `/workspace/outputs/` (checkpoints, saved automatically)

---

## **6. Complete Workflow Example**

```bash
# 1. Build custom image
docker buildx build --platform linux/amd64 -t llamafactory-h100:v1.0 .

# 2. Push to Docker Hub
docker tag llamafactory-h100:v1.0 myuser/llamafactory-h100:v1.0
docker push myuser/llamafactory-h100:v1.0

# 3. Add Docker Hub registry to OVHcloud
ovhai registry add index.docker.io

# 4. Upload config and data to Object Storage
# (via OVHcloud Control Panel or s3cmd)

# 5. Launch training job
ovhai job run index.docker.io/myuser/llamafactory-h100:v1.0 \
  --name "glm4-rust-hybrid-fp4-production" \
  --gpu 1 \
  --gpu-model H100 \
  --cpu 12 \
  --volume ai-training-data@GRA:/workspace/data:RO \
  --volume ai-checkpoints@GRA:/workspace/outputs:RW \
  --env WANDB_API_KEY=<your-key> \
  -- bash -c 'llamafactory-cli train /workspace/data/train_glm4_rust_v3_FINAL.yaml'

# 6. Monitor job
ovhai job logs <job-id> --follow
```

---

## **7. H100-Specific Optimizations**

Add to your Dockerfile for H100 compatibility:

```dockerfile
# Enable H100 Tensor Cores (Hopper architecture)
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV CUDA_VISIBLE_DEVICES="0"

# Optimize for H100 memory bandwidth
ENV NCCL_P2P_LEVEL=NVL
```

---

## **Key Differences from Standard Setup**

| Requirement | OVHcloud Specific | Standard Docker |
|-------------|-------------------|-----------------|
| User | **Must use UID 42420:42420** | Usually root or custom |
| HOME | **Must be /workspace** | Any directory |
| Architecture | **linux/amd64 only** | Platform-dependent |
| Registry | Shared/private required | Optional |
| Storage | Object Storage volumes | Local mounts |

---

## **Recommended Image for GLM-4.7 Project**

**Best Choice:** Custom Dockerfile based on `nvcr.io/nvidia/pytorch:24.08-py3`

**Why:**
- Native H100 support with CUDA 12.4
- Optimized NCCL for multi-GPU (if scaling later)
- Clean base without unnecessary packages
- LLaMA-Factory + dependencies installed fresh

**Estimated Build Time:** 5-10 minutes  
**Final Image Size:** ~15-20 GB (pushed to registry)
