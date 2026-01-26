# OVHcloud AI Training - LLaMA-Factory Custom Image
# Authorization: TPM-BMAD-2026-Q1-GO
# Build: docker buildx build --platform linux/amd64 -t llamafactory-h100:v1.0 .

FROM nvcr.io/nvidia/pytorch:24.08-py3

# CRITICAL: Set /workspace as HOME (OVHcloud requirement)
WORKDIR /workspace
ENV HOME=/workspace

# H100-specific optimizations
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV CUDA_VISIBLE_DEVICES="0"
ENV NCCL_P2P_LEVEL=NVL

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    screen \
    bc \
    && rm -rf /var/lib/apt/lists/*

# Install LLaMA-Factory dependencies
RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    peft>=0.10.0 \
    bitsandbytes>=0.43.0 \
    accelerate>=0.28.0 \
    datasets>=2.18.0 \
    sentencepiece \
    protobuf \
    tiktoken \
    wandb \
    scipy

# Clone and install LLaMA-Factory
RUN git clone https://github.com/hiyouga/LLaMA-Factory.git /workspace/LLaMA-Factory && \
    cd /workspace/LLaMA-Factory && \
    pip install -e .

# Try to install Unsloth (optional, may fail on some systems)
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" || echo "Unsloth install skipped"

# Copy training mission files
COPY config/ /workspace/config/
COPY scripts/ /workspace/scripts/
COPY launch_training.sh /workspace/
COPY monitor_heartbeat.sh /workspace/

# Make scripts executable
RUN chmod +x /workspace/*.sh /workspace/scripts/*.py

# CRITICAL: Give OVHcloud user (42420:42420) full access
RUN chown -R 42420:42420 /workspace

# Default command
CMD ["bash"]
