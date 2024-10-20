import re, time
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from utils import (
    init_gpu,
    show_gpu_utilization,
    path_checker,
    call_prompt_template,
    show_dataset_info,
    split_llama2_prompt,
    remove_repeated_sequences,
)

################################################################

# Base Model
base_model = "Mistral-Instruct"
base_model_name = "Mistral-7B-Instruct-v0.2"#"Mixtral-8x7B-Instruct-v0.1" #"Mistral-7B-Instruct-v0.2"
shared_model_dir = f"../shared_model/{base_model}"
base_model_dir = f"{shared_model_dir}/{base_model_name}"
kv_seqs = 128 # recommend 64 or less for 70B Model

# Quantization
# use only if run quantization # suggested "*-awq" model.
quantization_method = None # 'awq' | 'squeezellm' | None
if quantization_method :
    base_model += f"-{quantization_method}"
    base_model_name += f"-{quantization_method}"

# Adapter
use_lora = True
dataset_name = "PRM800K"
data_phase = 2
checkpoint = "800"
peft_model_dir = f"./models/{base_model_name}/Adapter-{dataset_name}-cp{checkpoint}"
#peft_model_dir = f"{shared_model_dir}/Adapter-{dataset_name}_phase{data_phase}-cp{checkpoint}"
lora_id = None

# Testset
#testset_path = "data/WAPLMATH/wapl_math_test.jsonl"
testset_path = f"data/{dataset_name}/test.jsonl"
test_idxs = [0,5,22]

# Pathes
PATHES = [base_model_dir, testset_path]
if use_lora:
    PATHES.append(peft_model_dir)
    lora_id = 1
path_checker(PATHES)

###############################
## Prompt ##
prompt_style = "llama2"
instruction_style = "problem_answer"
input_template, prompt_template= call_prompt_template(prompt_style, instruction_style)

################################################################

def print_inf_settings():
    print("[ Inference Settings ]")
    print(f" * Model: {base_model_name}")
    if quantization_method:
        print(f" * Quantizatione: {quantization_method}")
    if use_lora :
        print(" * Low-Lank Adaptation: ", peft_model_dir.split("/")[-1], "\n")

def deploy_llm():
    # Set Model
    start_loading_time = time.time()

    # set llm
    llm = LLM( 
        base_model_dir,
        tensor_parallel_size=2, # required GPU
        max_model_len = 4096, # == max seq len 32768; larger than 'KV cache' 26016 (our gpu..)
        
        # For OOM
        #gpu_memory_utilization=0.9 # default=0.9,
        max_num_seqs = kv_seqs, # default=256

        enable_lora=use_lora,
        max_loras=2, # default=1
        max_lora_rank = 64, # default == 16
        #worker_use_ray=True, default=false

        quantization=quantization_method, #'awq', # 'awq' | 'squeezellm' | None
    )
    show_gpu_utilization()
    
    end_loading_time = time.time()
    loading_time = end_loading_time - start_loading_time
    print(f"Time Taken for Load: \n {loading_time:.4f} seconds")
    return llm

def gen_instruction_dict(example):
    return {"text": input_template.format(example["problem"], example["answer"])}

if __name__ == "__main__":
    # Start
    print_inf_settings()
    # Clear GPU
    init_gpu()

    # testset
    test_dataset = load_dataset(
        "json",
        data_files={"test":testset_path},
        split='test'
    )
    show_dataset_info(test_dataset, "Check origin Test Dataset")
    
    # Use this when select cases
    if test_dataset:
        test_dataset=test_dataset.select(test_idxs)
    
    # Generate prompts
    testset = test_dataset.map(gen_instruction_dict)
    testset = testset.select_columns("text")
    show_dataset_info(testset, "Check Test Instructions")

    sampling_params = SamplingParams(
        #n=2, # num of return sequence; samples
        temperature=0.6, # Sharpness of Softmax distribution for Consistency
        max_tokens=2048, # defalt = 16 ...
        top_p=0.9,
        #top_k=50,
    )
    ## About 'temperature' 'and top_p'
    #    'temperature' is the randomness of the sampling
    #    'top_p' is the cumulative probability of the top tokens to consider
    #    'top_k' is the number of top tokens to consider
    #    Do not change more than two params
    pipe = deploy_llm()
    
    # Inference
    print("#" * 40, "Inference", "#" * 40)
    start_inference_time = time.time()
    
    outputs = pipe.generate(
            KeyDataset(testset, "text"),
            sampling_params,
            lora_request=LoRARequest("Adapter", lora_id, peft_model_dir)
            if lora_id else None )
    
    generated_texts = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        prompt = split_llama2_prompt(prompt)
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print("-"*20, f"<{i+1}'th request>","-"*20)
        #print(f"Prompt: {prompt!r}")
        print("Insturction: ", prompt[0].split("\n\n")[1])
        print("Generated text: \n", generated_text)
        print("Generated text(no repeat): \n", remove_repeated_sequences(generated_text))
    end_inference_time = time.time()
    inference_time = end_inference_time - start_inference_time
    print(f"Time Taken for Inference: \n {inference_time:.4f} seconds")