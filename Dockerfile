FROM runpod/worker-comfyui:5.7.1-base

# Install PuLID custom node (balazik version - face identity for FLUX)
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/balazik/ComfyUI-PuLID-Flux.git

# Install Python dependencies for PuLID + InsightFace
RUN pip install facexlib insightface onnxruntime-gpu ftfy timm sentencepiece

# Apply PuLID compatibility patches
COPY patch_pulid.py /tmp/patch_pulid.py
RUN python /tmp/patch_pulid.py

# Override model paths to match our network volume layout
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
