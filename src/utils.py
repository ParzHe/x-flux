import os
import pynvml
import torch
import gc
import json

def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024

def get_gpu_mem_info(gpu_id=0):
    """
    根据显卡 id 获取显存使用信息, 单位 MB
    :param gpu_id: 显卡 ID
    :return: total 所有的显存，used 当前使用的显存, free 可使用的显存
    """
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(f"gpu_id {gpu_id} 对应的显卡不存在!")
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(bytes_to_giga_bytes(meminfo.total), 2)
    used = round(bytes_to_giga_bytes(meminfo.used))
    free = round(bytes_to_giga_bytes(meminfo.free), 2) # GB
    return total, used, free

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

def flush_without_peak():
    gc.collect()
    torch.cuda.empty_cache()
    # torch.cuda.reset_max_memory_allocated()
    # torch.cuda.reset_peak_memory_stats()

def save_images(images,timestamp,output_folder):  
    output_folder = output_folder
    onetime_output_folder = os.path.join(output_folder,timestamp)
    os.makedirs(onetime_output_folder, exist_ok=True)
    saved_paths = []
    
    for i, img in enumerate(images):
        filename = f"output_{i}.png"
        filepath = os.path.join(onetime_output_folder, filename)
        img.save(filepath)
        saved_paths.append(filepath)
    
    return images,saved_paths

def save_images_with_prompt(
    prompt=" ",
    checkpoint="",
    seed=42,
    guidance_scale=0.0,
    width=1024, height=1024,
    num_inference_steps=4,
    max_memory_usage=0.0,
    generation_time=0.0,
    images=None,
    timestamp=None,
    output_folder: str = None,
    # filename='diffusion_params.json'
):
    """
    Saves the parameters used in generating an image with a diffuser model to a JSON file.

    Args:
        prompt (str): The text prompt used for generation.
        checkpoint (str): The checkpoint name or path of the model.
        seed (int): The random seed for reproducibility.
        guidance_scale (float): The scale for classifier-free guidance.
        width (int): The width of the generated image.
        height (int): The height of the generated image.
        num_inference_steps (int): The number of inference steps.
        max_memory_usage (float): Maximum GPU memory usage during generation (in MB).
        generation_time (float): Time taken to generate the image (in seconds).
        output_folder (str): The folder of the output files.

    Returns:
        iamges (PIL): The images data.
        save_paths (sta): Saved paths of the iamges
    """
    
    params = {
        'prompt': prompt,
        'checkpoint': checkpoint,
        'seed': seed,
        'guidance_scale': guidance_scale,
        'width': width,
        'height': height,
        'num_inference_steps': num_inference_steps,
        'max_memory_usage': f"{max_memory_usage} GB",
        'generation_time': f"{generation_time} s",
    }
    
    output_folder = output_folder
    onetime_output_folder=os.path.join(output_folder,str(timestamp))
    os.makedirs(onetime_output_folder, exist_ok=True)
    
    filename = f"prompt_{timestamp}.json"
    filepath=os.path.join(onetime_output_folder,filename)

    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4)
    
    return save_images(images,timestamp)