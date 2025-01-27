# Stage 1: Base image with common dependencies
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

ARG GITHUB_TOKEN

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1 \
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
RUN . venv/bin/activate && /usr/bin/yes | comfy --workspace /comfyui install --cuda-version 12.1 --nvidia --version 0.3.6


# Change working directory to ComfyUI
WORKDIR /comfyui

# Install runpod
RUN . ../venv/bin/activate && pip install runpod requests

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /
# Optionally copy the snapshot file
ADD worker_snapshot.json /

# Add scripts
ADD src/start.sh src/restore_snapshot.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh /restore_snapshot.sh && . venv/bin/activate && /restore_snapshot.sh


RUN . venv/bin/activate && pip install --upgrade typing_extensions

RUN . venv/bin/activate && pip install torchvision

RUN . venv/bin/activate && comfy update all && comfy update comfy

RUN . venv/bin/activate && pip install xformers


# Start container
CMD ["/start.sh"]

# Stage 2: Download models
FROM base AS downloader

SHELL ["/bin/bash", "-c"]

ARG HUGGINGFACE_ACCESS_TOKEN
RUN echo "Token prefix: ${HUGGINGFACE_ACCESS_TOKEN:0:8}..."

# Change working directory to ComfyUI
WORKDIR /comfyui

# Create necessary directories
RUN mkdir -p models/checkpoints models/vae models/loras models/style_models models/clip_vision models/unet models/clip input


# Download checkpoints/vae/LoRA to include in image based on model type
# RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O /comfyui/models/unet/flux1-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors && \
#       wget -O /comfyui/models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
#       wget -O /comfyui/models/clip/t5xxl_fp16.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
#       wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O /comfyui/models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors && \
#       wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O /comfyui/models/clip_vision/sigclip_vision_patch14_384.safetensors https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors && \
#       wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O /comfyui/models/style_models/flux1-redux-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev/resolve/main/flux1-redux-dev.safetensors




# Stage 3: Final image
FROM base AS final

# Add Your Own Models and Files
COPY models/ChrismasSuit.png /comfyui/input/
COPY models/elf.png /comfyui/input/
COPY models/padoru.png /comfyui/input/
COPY models/reindeer.png /comfyui/input/
COPY models/santa.png /comfyui/input/


COPY models/MooDeng.safetensors /comfyui/models/loras/
RUN ls /comfyui/models/loras/
RUN mkdir -p /comfyui/models/LLM /comfyui/models/sams /comfyui/models/grounding-dino

# Copy models from the docker image
COPY --from=whitemoney293/comfyui-flux-models:v1.1.0 /models/unet/flux1-dev.safetensors /comfyui/models/unet/
COPY --from=whitemoney293/comfyui-flux-models:v1.1.0 /models/vae/ae.safetensors /comfyui/models/vae/
COPY --from=whitemoney293/comfyui-flux-models:v1.1.0 /models/clip_vision/sigclip_vision_patch14_384.safetensors /comfyui/models/clip_vision/
COPY --from=whitemoney293/comfyui-flux-models:v1.1.0 /models/style_models/flux1-redux-dev.safetensors /comfyui/models/style_models/
COPY --from=whitemoney293/comfyui-flux-models:v1.1.0 /models/clip/clip_l.safetensors /comfyui/models/clip/
COPY --from=whitemoney293/comfyui-flux-models:v1.1.0 /models/clip/t5xxl_fp8_e4m3fn.safetensors /comfyui/models/clip/
COPY --from=whitemoney293/comfyui-flux-models:v1.1.0 /models/unet/flux1-fill-dev.safetensors /comfyui/models/unet/
COPY --from=whitemoney293/comfyui-flux-models:v1.1.0 /models/clip_vision/siglip-so400m-patch14-384.safetensors /comfyui/models/clip_vision/
COPY --from=whitemoney293/comfyui-flux-models:v1.1.0 /models/clip/ViT-L-14-BEST-smooth-GmP-ft.safetensors /comfyui/models/clip/
COPY --from=whitemoney293/comfyui-flux-models:v1.1.0 /models/unet/flux1-canny-dev.safetensors /comfyui/models/unet/

# downloading https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth to /comfyui/models/sams/sam_vit_h_4b8939.pth
RUN wget -O /comfyui/models/sams/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# downloading https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py to /comfyui/models/grounding-dino/GroundingDINO_SwinT_OGC.cfg.py
RUN wget -O /comfyui/models/grounding-dino/GroundingDINO_SwinT_OGC.cfg.py https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py

# downloading https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth to /comfyui/models/grounding-dino/groundingdino_swint_ogc.pth
RUN wget -O /comfyui/models/grounding-dino/groundingdino_swint_ogc.pth https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth

# Copy models from stage 2 to the final image

# COPY --from=downloader /comfyui/models /comfyui/models
# COPY --from=downloader /comfyui/input /comfyui/input

# COPY models/flux1-dev.safetensors models/unet/
# COPY models/clip_l.safetensors models/clip/
# COPY models/t5xxl_fp16.safetensors models/clip/
# COPY models/ae.safetensors models/vae/
# COPY models/sigclip_vision_patch14_384.safetensors models/clip_vision/
# COPY models/flux1-redux-dev.safetensors models/style_models/



# Check if models are downloaded
RUN echo "Input files:" && ls /comfyui/input/ && \
    echo "Lora files:" && ls /comfyui/models/loras/ && \
    echo "Unet files:" && ls /comfyui/models/unet/ && \
    echo "VAE files:" && ls /comfyui/models/vae/ && \
    echo "Clip files:" && ls /comfyui/models/clip/ && \
    echo "Clip vision files:" && ls /comfyui/models/clip_vision/ && \
    echo "Style models files:" && ls /comfyui/models/style_models

RUN . /venv/bin/activate && comfy env




CMD ["/start.sh"]