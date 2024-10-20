import safetensors.torch
import os, re

def safetensor_process(dirpath, filename):

    tensors =  safetensors.torch.load_file(os.path.join(dirpath, filename))

    nonlora_keys = []
    for k in list(tensors.keys()):
        if "lora" not in k:
            nonlora_keys.append(k)
    # print(nonlora_keys) # just take a look what they are

    for k in nonlora_keys:
        del tensors[k]

    safetensors.torch.save_file(tensors, os.path.join(dirpath, 'new_'+filename))
    print(f'saved at {dirpath}')

def vllm_lora_process(root_dir):
    pattern = re.compile(r'.safetensors')
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if pattern.search(filename):

                safetensor_process(dirpath, filename)

if __name__ == "__main__":
    vllm_lora_process("./Llama-3-8B-Instruct-awq")
    print("preprocessing is done.")