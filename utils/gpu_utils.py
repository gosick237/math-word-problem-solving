import sys
import torch
import nvidia_smi
import torch.version

def show_cuda_gpus():
    print('## Version ##')
    print('  Python VERSION: ', sys.version)
    print('  Pytorch VERSION: ', torch.__version__)
    print('  CUDA VERSION: ', torch.version.cuda)
    print('  CUDNN VERSION: ', torch.backends.cudnn.version())

    print('## CUDA Details ##')
    print('TORCH_CUDA: ', torch.cuda.is_available())
    print('Number CUDA Devices:', torch.cuda.device_count())
    print ('Current cuda device: ', torch.cuda.current_device(), ' **May not correspond to nvidia-smi ID above, check visibility parameter')
    print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))


def init_gpu():
    print('*'*30,'Resource Check' ,'*'*30)
    mem_flush()
    show_gpu_utilization()
    print('*'*70)

def bfloat16_available():
    # Compatibility with bfloat16
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("* Your GPU supports bfloat16: accelerate training with bf16=True")

def mem_flush():
    #device = cuda.get_current_device()
    #device.reset()
    torch.cuda.empty_cache()

def show_gpu_utilization():
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()

    print(' [Memory Usage] :')
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print(f"    [GPU-{i}] : {info.used//1024**2} / {info.total//1024**2} MB.")
    nvidia_smi.nvmlShutdown()