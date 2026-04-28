# Base: CUDA 13.1 + cuDNN, targeting RTX 3060 (Ampere sm_86, max CUDA 13.1).
# cudnn-devel: needed for cuDNN headers when face_detection/face_alignment build from source,
# and for cuDNN runtime used by PyTorch (cudnn.benchmark = True in workers).
# PyTorch cu128 wheels are ABI-compatible with CUDA 13.x at runtime.
FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility

# Target Ampere sm_86 explicitly so nvcc only compiles for this architecture.
# Speeds up any JIT compilation and avoids generating unused PTX/SASS for older arches.
ENV TORCH_CUDA_ARCH_LIST="8.6"

# Giảm fragmentation cho cấp phát CUDA của PyTorch.
# expandable_segments=True cho phép caching allocator nới rộng segment hiện có
# thay vì giữ nguyên các block đã reserve nhưng không cấp phát được — giải pháp
# trực tiếp cho lỗi "1.48 GiB reserved by PyTorch but unallocated" trên GPU 12 GB.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OS deps tối thiểu (đã loại nasm/yasm/x264-dev/x265-dev — không còn build FFmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl gnupg ca-certificates \
        build-essential pkg-config \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
        libgl1-mesa-glx \
        libsndfile1 libportaudio2 \
        python3.10 python3.10-dev python3-pip \
        gcc \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 \
    && rm -rf /var/lib/apt/lists/*

# FFmpeg 7 với NVENC/NVDEC từ Jellyfin APT repo (~30 giây, thay vì ~15 phút build từ source).
# Jellyfin-ffmpeg7 đã build với --enable-nvenc --enable-libx264 --enable-libx265 --enable-cuvid.
# `apt-key` không dùng nữa trên Ubuntu 22.04 → key bỏ vào /usr/share/keyrings.
RUN curl -fsSL https://repo.jellyfin.org/jellyfin_team.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/jellyfin.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/jellyfin.gpg] \
              https://repo.jellyfin.org/ubuntu jammy main" \
        > /etc/apt/sources.list.d/jellyfin.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends jellyfin-ffmpeg7 \
    && ln -sf /usr/lib/jellyfin-ffmpeg/ffmpeg  /usr/local/bin/ffmpeg \
    && ln -sf /usr/lib/jellyfin-ffmpeg/ffprobe /usr/local/bin/ffprobe \
    && echo "/usr/lib/jellyfin-ffmpeg/lib" > /etc/ld.so.conf.d/jellyfin-ffmpeg.conf \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /app/resume_state /app/vnlr /app/labels /app/dataset
# PyTorch stack — cu128 wheels support Ampere sm_86 (RTX 3060) and later.
# Không pin version: pip resolver tự chọn bản torch + torchcodec mới nhất trong cu128.
# Đảm bảo torch + torchvision + torchaudio + torchcodec luôn khớp ABI.
RUN pip install --no-cache-dir --timeout 300 --retries 5 --upgrade pip wheel setuptools

RUN pip install --no-cache-dir --timeout 600 --retries 5 \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch torchvision torchaudio torchcodec

# Các Python deps còn lại (đã loại bỏ torch* và triton khỏi requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 --retries 5 -r requirements.txt

# face_detection + face_alignment (Imperial College, hhj1897)
RUN git clone https://github.com/hhj1897/face_detection.git /tmp/face_detection \
    && pip install -e /tmp/face_detection

RUN git clone https://github.com/hhj1897/face_alignment.git /tmp/face_alignment \
    && pip install -e /tmp/face_alignment

# Copy project code
COPY . .
