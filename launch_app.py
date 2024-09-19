import os
import torch

from src.utils import get_gpu_mem_info

if __name__ == "__main__":
    
    gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info(gpu_id=0)
    cmd = ""
    
    torch.cuda.empty_cache()
    
    if gpu_mem_total > 35:
        cmd="python gradio_demo.py --port 1024 --name flux-dev --ckpt_dir ../models --output_dir ../fssd/output --pipeline xflux"
    else: 
        print(f"所选显卡显存总共为{gpu_mem_total} GB。其不足以载入非量化的Flux-dev 和 Flux-schnell 所需的所有模型在GPU中，自动以低显存模式启动...")
        cmd="python gradio_demo.py --port 1024 --name flux-dev --offload --ckpt_dir ../models --output_dir ../fssd/output --pipeline diffusers"
    
    os.system(cmd)