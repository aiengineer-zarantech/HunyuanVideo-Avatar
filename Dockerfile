# 🚀 CUDA 12.8 Runtime (RTX 5090 / Blackwell support ke liye)
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# ✅ 基本環境設定
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1

# 🧩 安裝必要套件
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ffmpeg \
    curl \
    libsm6 \
    libxext6 \
    tzdata \
 && rm -rf /var/lib/apt/lists/*

# ⚙️ 設定工作目錄
WORKDIR /workspace
COPY . /workspace

# 🐍 安裝 PyTorch GPU 對應版本 (CUDA 12.8 - RTX 5090 ke liye)
# Note: Base image mein pehle se torch hai, lekin ensure karne ke liye explicitly install karein
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 📦 安裝模型依賴
RUN pip install --prefer-binary -r requirements.txt
RUN pip install --no-cache-dir runpod==1.4.0 requests==2.31.0

# ⚡ FlashAttention Dummy (same as before)
RUN mkdir -p /workspace/flash_attn && \
    echo "def flash_attn_func(*args, **kwargs):\n    raise NotImplementedError('FlashAttention disabled.')\n\n" > /workspace/flash_attn/__init__.py && \
    echo "def flash_attn_varlen_func(*args, **kwargs):\n    raise NotImplementedError('FlashAttention disabled.')\n\n" >> /workspace/flash_attn/__init__.py && \
    echo "def flash_attn_unpadded_func(*args, **kwargs):\n    raise NotImplementedError('FlashAttention disabled.')\n\n" >> /workspace/flash_attn/__init__.py && \
    mkdir -p /workspace/flash_attn/flash_attn_interface && \
    echo 'def flash_attn_varlen_func(*a, **k): raise NotImplementedError("FlashAttention disabled")' > /workspace/flash_attn/flash_attn_interface/__init__.py

# 🌐 RunPod Serverless port
EXPOSE 5000

# 🚦 Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
 CMD curl -f http://localhost:5000/ || exit 1

# ▶️ Start handler
CMD ["python", "handler.py"]
