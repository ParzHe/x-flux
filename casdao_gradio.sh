#!/bin/bash

export GRADIO_TEMP_DIR="/home/tom/fssd/tmp/"
export HUGGINGFACE_HUB_CACHE="/home/tom/fssd/HF_cache"
export HF_HOME="home/tom/fssd/HF"

export FLUX_DEV="../fshare/models/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors"
export FLUX_DEV_FP8="../fshare/models/XLabs-AI/flux-dev-fp8/flux-dev-fp8.safetensors"
export FLUX_SCHNELL="../fshare/models/black-forest-labs/FLUX.1-schnell/flux1-schnell.safetensors"
export AE="../fshare/models/black-forest-labs/FLUX.1-dev/ae.safetensors"

mkdir -p fssd/models

cp -n fshare/models/XLabs-AI/flux-lora-collection/anime_lora.safetensors fssd/models/anime_lora.safetensors
cp -n fshare/models/XLabs-AI/flux-lora-collection/art_lora.safetensors fssd/models/art_lora.safetensors
cp -n fshare/models/XLabs-AI/flux-lora-collection/disney_lora.safetensors fssd/models/disney_lora.safetensors
cp -n fshare/models/XLabs-AI/flux-lora-collection/furry_lora.safetensors fssd/models/furry_lora.safetensors
cp -n fshare/models/XLabs-AI/flux-lora-collection/mjv6_lora.safetensors fssd/models/mjv6_lora.safetensors
cp -n fshare/models/XLabs-AI/flux-lora-collection/realism_lora.safetensors fssd/models/realism_lora.safetensors
cp -n fshare/models/XLabs-AI/flux-lora-collection/scenery_lora.safetensors fssd/models/scenery_lora.safetensors

cp -n fshare/models/XLabs-AI/flux-controlnet-collections/flux-canny-controlnet-v3.safetensors fssd/models/flux-canny-controlnet-v3.safetensors
cp -n fshare/models/XLabs-AI/flux-controlnet-collections/flux-depth-controlnet-v3.safetensors fssd/models/flux-depth-controlnet-v3.safetensors
cp -n fshare/models/XLabs-AI/flux-controlnet-collections/flux-hed-controlnet-v3.safetensors fssd/models/flux-hed-controlnet-v3.safetensors

cp -n fshare/models/XLabs-AI/flux-ip-adapter/flux-ip-adapter.safetensors fssd/models/flux-ip-adapter.safetensors

cd x-flux

python gradio_demo.py --port 1024 --name flux-dev-fp8 --offload --ckpt_dir ../fssd/models --output_dir ../fssd/output
