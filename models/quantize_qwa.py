from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import os, nvidia_smi

## Model & Data Path ##
# Base Model
base_model = "Mixtral-Instruct"
base_model_name = "Mixtral-8x7B-Instruct-v0.1"
base_model_dir = f"../shared_model/{base_model}/{base_model_name}"
qt_model_dir = base_model_dir + "-awq-new"

if not os.path.exists(base_model_dir):
    print(f"Given path '{base_model_dir}' is not exist. Check your path!")
    quit()

device_map = "auto"

quantization_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

def show_gpu_utilization():
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()

    print('  [Memory Usage] :')
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print(f"\tGPU{i} memory occupied: {info.used//1024**2} MB.")
    nvidia_smi.nvmlShutdown()

if __name__ == "__main__":
    # Load model
    model = AutoAWQForCausalLM.from_pretrained(
        base_model_dir,
        device_map=device_map,
        local_files_only=True,
        **{"low_cpu_mem_usage": True, "use_cache": False}
    )
    tokenizer=AutoTokenizer.from_pretrained(base_model_dir, local_files_only=True)
    print("Memory After Model load:")
    show_gpu_utilization()

    # Quantize
    model.quantize(tokenizer, quant_config=quantization_config)

    # Save quantized model
    model.save_quantized(qt_model_dir)
    tokenizer.save_pretrained(qt_model_dir)

    print(f'Model is quantized and saved at "{qt_model_dir}"')