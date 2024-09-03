#!/bin/bash

export GRADIO_TEMP_DIR="/home/tom/fssd/tmp/"
export HUGGINGFACE_HUB_CACHE="/home/tom/fssd/HF_cache"
export HF_HOME="home/tom/fssd/HF"

export FLUX_DEV="../fshare/models/black-forest-labs/FLUX.1-dev"
export FLUX_DEV_FP8="../fshare/models/XLabs-AI/flux-dev-fp8"
export FLUX_SCHNELL="../fshare/models/black-forest-labs/FLUX.1-schnell"
export AE="../fshare/models/black-forest-labs/FLUX.1-dev/ae.safetensors"

cd x-flux

python gradio_demo.py --port 1024 --name flux-dev-fp8 --ckpt_dir /home/tom/fshare/models/XLabs-AI --offload
