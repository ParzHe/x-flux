import os
import torch

from src.utils import get_gpu_mem_info

if __name__ == "__main__":
    
    gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info(gpu_id=0)
    cmd = ""
    
    torch.cuda.empty_cache()
    
    if gpu_mem_total > 35:
        cmd="python gradio_demo.py --port 1024 --name flux-dev --ckpt_dir ../fssd/models --output_dir ../fssd/output"
    else: 
        cmd="pyhton gradio_demo.py --port 1024 --name flux-dev --offload --ckpt_dir ../fssd/models --output_dir ../fssd/output"
    os.system(cmd)