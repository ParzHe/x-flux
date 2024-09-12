import pynvml

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
    total = round(meminfo.total / 1024 / 1024 / 1024, 2) # GB
    used = round(meminfo.used / 1024 / 1024 / 1024, 2) # GB
    free = round(meminfo.free / 1024 / 1024 / 1024, 2) # GB
    return total, used, free