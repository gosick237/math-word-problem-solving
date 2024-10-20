from awq import AutoAWQForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import os, nvidia_smi, torch

## Model & Data Path ##
# Base Model
base_model = "Mixtral-Instruct"
base_model_name = "Mixtral-8x7B-Instruct-v0.1"
base_model_dir = f"/workspace/shared_model/{base_model}/{base_model_name}"
qt_model_dir = base_model_dir + "-bnb-new"

if not os.path.exists(base_model_dir):
    print(f"Given path '{base_model_dir}' is not exist. Check your path!")
    quit()

device_map = "auto"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, # Activate 4-bit precision base model loading
    bnb_4bit_quant_type="nf4", # fp4 or nf4
    bnb_4bit_compute_dtype=torch.bfloat16, # "float16",
    bnb_4bit_use_double_quant=True, # double quantization
)

def show_gpu_utilization():
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()

    print('  [Memory Usage] :')
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print(f"\tGPU{i} memory occupied: {info.used//1024**2} MB.")
    nvidia_smi.nvmlShutdown()

######################
######################

if __name__ == "__main__":
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        quantization_config=quantization_config,
        device_map=device_map,
        local_files_only=True,
        **{"low_cpu_mem_usage": True, "use_cache": False}
    )
    tokenizer=AutoTokenizer.from_pretrained(base_model_dir, local_files_only=True)
    print("Memory After Model load:")
    show_gpu_utilization()

    # Save quantized model
    model.save_pretrained(qt_model_dir)
    tokenizer.save_pretrained(qt_model_dir)

    print(f'Model is quantized and saved at "{qt_model_dir}"')