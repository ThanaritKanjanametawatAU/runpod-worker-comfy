# Stage 1: Base image with common dependencies
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8




# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1 \
    ffmpeg \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y python3.10-venv

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

RUN python3 -m venv venv

RUN . venv/bin/activate && pip install --upgrade pip && which python \
 && python --version

# Install comfy-cli
RUN . venv/bin/activate && pip install comfy-cli

# Install ComfyUI
# RUN /usr/bin/yes | comfy --workspace /comfyui install --cuda-version 12.1 --nvidia --version 0.3.4
RUN . venv/bin/activate && /usr/bin/yes | comfy --workspace /comfyui install --cuda-version 12.4 --nvidia


# Change working directory to ComfyUI
WORKDIR /comfyui

# Install runpod
RUN . ../venv/bin/activate && pip install runpod requests cloudinary

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /
# Optionally copy the snapshot file
ADD worker_snapshot.json /

# Add scripts and Restore snapshot
ADD src/start.sh src/restore_snapshot.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh /restore_snapshot.sh && . venv/bin/activate && /restore_snapshot.sh

# Clone ComfyUI-WanVideoWrapper custom node
RUN mkdir -p /comfyui/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper /comfyui/custom_nodes/ComfyUI-WanVideoWrapper


RUN . venv/bin/activate && pip install --upgrade typing_extensions


RUN . venv/bin/activate && comfy update all

# Install torch, torchvision, torchaudio, and xformers first (all pinned, using the correct index URL)
RUN . venv/bin/activate && pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install the compatible xformers version for torch 2.6.0 and CUDA 12.4
RUN . venv/bin/activate && pip install xformers==0.0.29.post2 --index-url https://download.pytorch.org/whl/cu124

# Install build tools and ninja before Python dependencies
RUN apt-get update && apt-get install -y build-essential python3-dev && \
    . venv/bin/activate && pip install --upgrade pip && pip install ninja packaging

# Install the rest of your dependencies (without flash-attn)
RUN . venv/bin/activate && pip install --upgrade huggingface_hub==0.30.2 diffusers transformers transparent_background tiktoken moviepy triton setuptools wheel psutil --no-build-isolation

# Install flash-attn separately with no build isolation and limited jobs
RUN . venv/bin/activate && MAX_JOBS=4 pip install flash-attn --no-build-isolation

# Install custom node requirements without allowing torch upgrades
RUN . venv/bin/activate && pip install --no-deps -r /comfyui/custom_nodes/ComfyUI-SkyReels-A2/requirements.txt

# Stage 3: Final image
FROM base AS final

# Add Your Own Models and Files
# COPY models/ChrismasSuit.png /comfyui/input/
# COPY models/elf.png /comfyui/input/
# COPY models/padoru.png /comfyui/input/
# COPY models/reindeer.png /comfyui/input/
# COPY models/santa.png /comfyui/input/

# Download Fish Speech models
# RUN mkdir -p /comfyui/models/checkpoints /comfyui/models/vae /comfyui/models/loras /comfyui/models/style_models \
# /comfyui/models/clip_vision /comfyui/models/unet /comfyui/models/clip /comfyui/input \
# /comfyui/models/sonic /comfyui/custom_nodes/ComfyUI_FishSpeech_EX_PP/checkpoints/fish-speech-1.5/ \
# /comfyui/custom_nodes/comfyui_layerstyle/RMBG-1.4

# Use BuildKit secret mount to securely access the token during build
# RUN --mount=type=secret,id=huggingface,target=/run/secrets/huggingface \
#     HUGGINGFACE_TOKEN=$(cat /run/secrets/huggingface) \
#     wget --header="Authorization: Bearer $HUGGINGFACE_TOKEN" -O/comfyui/custom_nodes/ComfyUI_FishSpeech_EX_PP/checkpoints/fish-speech-1.5/model.pth \
#     https://huggingface.co/fishaudio/fish-speech-1.5/resolve/main/model.pth

# RUN --mount=type=secret,id=huggingface,target=/run/secrets/huggingface \
#     HUGGINGFACE_TOKEN=$(cat /run/secrets/huggingface) \
#     wget --header="Authorization: Bearer $HUGGINGFACE_TOKEN" -O /comfyui/custom_nodes/ComfyUI_FishSpeech_EX_PP/checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth \
#     https://huggingface.co/fishaudio/fish-speech-1.5/resolve/main/firefly-gan-vq-fsq-8x1024-21hz-generator.pth


RUN --mount=type=secret,id=huggingface,target=/run/secrets/huggingface \
    HUGGINGFACE_TOKEN=$(cat /run/secrets/huggingface) && \
    . venv/bin/activate && \
    HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN huggingface-cli download fishaudio/fish-speech-1.5 \
    --include model.pth \
    --local-dir /comfyui/custom_nodes/ComfyUI_FishSpeech_EX_PP/checkpoints/fish-speech-1.5/

RUN --mount=type=secret,id=huggingface,target=/run/secrets/huggingface \
    HUGGINGFACE_TOKEN=$(cat /run/secrets/huggingface) && \
    . venv/bin/activate && \
    HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN huggingface-cli download fishaudio/fish-speech-1.5 \
    --include firefly-gan-vq-fsq-8x1024-21hz-generator.pth \
    --local-dir /comfyui/custom_nodes/ComfyUI_FishSpeech_EX_PP/checkpoints/fish-speech-1.5/

RUN --mount=type=secret,id=huggingface,target=/run/secrets/huggingface \
    HUGGINGFACE_TOKEN=$(cat /run/secrets/huggingface) && \
    . venv/bin/activate && \
    HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN huggingface-cli download briaai/RMBG-1.4 \
    --include model.pth \
    --local-dir /comfyui/custom_nodes/comfyui_layerstyle/RMBG-1.4/


COPY models/*.png /comfyui/input/
# COPY models/MooDeng.safetensors /comfyui/models/loras/
# COPY models/sonic/ /comfyui/models/sonic/


# Check if models are downloaded
RUN apt-get update && apt-get install -y tree && \
    echo "Verifying models directory structure:" && \
    tree /comfyui/models && \
    # Cleanup apt cache to reduce image size
    rm -rf /var/lib/apt/lists/*


RUN . /venv/bin/activate && comfy env

RUN rm -rf /comfyui/models && ln -s /runpod-volume/models /comfyui/models

CMD ["/start.sh"]