import re, time
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset, concatenate_datasets
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from utils import (
    init_gpu,
    show_gpu_utilization,
    path_checker,
    call_prompt_template,
    show_dataset_info,
    split_with_pattern,
    get_completion_with_ans
)

################################################################

# Base Model
base_model = "Llama3-8B-Instruct-awq"
base_model_name = "Llama-3-8B-Instruct-awq"
shared_model_dir = f"../shared_model/{base_model}"
base_model_dir = f"{shared_model_dir}/{base_model_name}"
kv_seqs = 256 # recommend 64 or less for 70B Model

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
checkpoint = "1200"
peft_model_dir = f"./models/{base_model_name}/Adapter-{dataset_name}-cp{checkpoint}"
#peft_model_dir = f"{shared_model_dir}/Adapter-{dataset_name}_phase{data_phase}-cp{checkpoint}"
lora_id = None

# Testset
#testset_path = "data/WAPLMATH/wapl_math_test.jsonl"
testset_path = f"data/{dataset_name}/test.jsonl"
test_idxs = [0,5,22,50]

# Pathes
PATHES = [base_model_dir, testset_path]
if use_lora:
    PATHES.append(peft_model_dir)
    lora_id = 1
path_checker(PATHES)

###############################
## Prompt ##
prompt_style = "llama3"
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
        temperature=0.7, # Sharpness of Softmax distribution for Consistency
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
        prompt = split_with_pattern(r"<\|begin_of_text\|><\|start_header_id\|>user<\|end_header_id\|>|<\|eot_id\|>",prompt)
        generated_text = output.outputs[0].text.strip()
        completion, answer = get_completion_with_ans(generated_text)
        generated_texts.append(generated_text)
        print("*"*20, f"<{i+1}'th request>","*"*20)
        #print(f"Prompt: {prompt!r}")
        print("Insturction: ", prompt[0])
        print("-"*30)
        print("Generated text (Origin): \n", generated_text)
        print("-"*30)
        print("Generated text (Processed): \n", completion)
        print("*"*60, "\n")
    end_inference_time = time.time()
    inference_time = end_inference_time - start_inference_time
    print(f"Time Taken for Inference: \n {inference_time:.4f} seconds")