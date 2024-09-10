#!/bin/bash

export GRADIO_TEMP_DIR="/home/tom/fssd/tmp/"
export HUGGINGFACE_HUB_CACHE="/home/tom/fssd/HF_cache"
export HF_HOME="home/tom/fssd/HF"

# export FLUX_DEV="../fshare/models/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors"
export FLUX_DEV="../fssd/models/ckpt/flux1-dev.safetensors"
export FLUX_DEV_FP8="../fshare/models/XLabs-AI/flux-dev-fp8/flux-dev-fp8.safetensors"
export FLUX_SCHNELL="../fshare/models/black-forest-labs/FLUX.1-schnell/flux1-schnell.safetensors"
export AE="../fshare/models/black-forest-labs/FLUX.1-dev/ae.safetensors"

mkdir -p fssd/models

mkdir -p fssd/models/ckpt
cp -n -v fshare/models/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors fssd/models/ckpt/flux1-dev.safetensors
# cp -n -v fshare/models/XLabs-AI/flux-dev-fp8/flux-dev-fp8.safetensors fssd/models/ckpt/flux-dev-fp8.safetensors
# cp -n -v fshare/models/black-forest-labs/FLUX.1-schnell/flux1-schnell.safetensors fssd/models/ckpt/flux1-schnell.safetensors

mkdir -p fssd/models/LoRA
cp -n -v fshare/models/XLabs-AI/flux-lora-collection/anime_lora.safetensors fssd/models/LoRA/anime_lora.safetensors
cp -n -v fshare/models/XLabs-AI/flux-lora-collection/art_lora.safetensors fssd/models/LoRA/art_lora.safetensors
cp -n -v fshare/models/XLabs-AI/flux-lora-collection/disney_lora.safetensors fssd/models/LoRA/disney_lora.safetensors
cp -n -v fshare/models/XLabs-AI/flux-lora-collection/furry_lora.safetensors fssd/models/LoRA/furry_lora.safetensors
cp -n -v fshare/models/XLabs-AI/flux-lora-collection/mjv6_lora.safetensors fssd/models/LoRA/mjv6_lora.safetensors
cp -n -v fshare/models/XLabs-AI/flux-lora-collection/realism_lora.safetensors fssd/models/LoRA/realism_lora.safetensors
cp -n -v fshare/models/XLabs-AI/flux-lora-collection/scenery_lora.safetensors fssd/models/LoRA/scenery_lora.safetensors
# cp -n -v fshare/models/Shakker-Labs/FLUX.1-dev-LoRA-collections/FLUX-dev-lora-Black_Myth_Wukong_hyperrealism_v1.safetensors fssd/models/LoRA/FLUX-dev-lora-Black_Myth_Wukong_hyperrealism_v1.safetensors

mkdir -p fssd/models/Controlnet
cp -n -v fshare/models/XLabs-AI/flux-controlnet-collections/flux-canny-controlnet-v3.safetensors fssd/models/Controlnet/flux-canny-controlnet-v3.safetensors
cp -n -v fshare/models/XLabs-AI/flux-controlnet-collections/flux-depth-controlnet-v3.safetensors fssd/models/Controlnet/flux-depth-controlnet-v3.safetensors
cp -n -v fshare/models/XLabs-AI/flux-controlnet-collections/flux-hed-controlnet-v3.safetensors fssd/models/Controlnet/flux-hed-controlnet-v3.safetensors

mkdir -p fssd/models/IP_Adapter
cp -n -v fshare/models/XLabs-AI/flux-ip-adapter/flux-ip-adapter.safetensors fssd/models/IP_Adapter/flux-ip-adapter.safetensors

cd x-flux

python gradio_demo.py --port 1024 --name flux-dev --offload --ckpt_dir ../fssd/models --output_dir ../fssd/output
