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
)

################################################################

# Base Model
base_model = "Komistral-Instruct"
base_model_name = "Komistral-7B-Instruct"#"Mixtral-8x7B-Instruct-v0.1" #"Mistral-7B-Instruct-v0.2"
shared_model_dir = f"../shared_model/{base_model}"
base_model_dir = f"{shared_model_dir}/{base_model_name}"
kv_seqs = 64 # recommend 64 for 70B Model; default = 256

# Quantization
# ** [01.APR.2024] LoRA is not supported with quantized models yet **
quantization_method = None # 'awq' | 'squeezellm' | None
if quantization_method :
    base_model_dir = base_model_dir + "-" + quantization_method
    base_model = base_model + "-" + quantization_method

# Adapter
use_lora = True
dataset_name = "WAPLMATH"
checkpoint = "841"
peft_model_dir = f"./models/Komistral-Instruct-woLMHead/Adapter-{dataset_name}-cp{checkpoint}"
#peft_model_dir = f"{shared_model_dir}/Adapter-{dataset_name}-cp{checkpoint}"
lora_id = None

# Testset
testset_path = f"data/{dataset_name}/wapl_math_test.jsonl"
test_idxs = [0,5,22]

# Pathes
PATHES = [base_model_dir, testset_path]
if use_lora:
    PATHES.append(peft_model_dir)
    lora_id = 1

def print_inf_settings():
    print("[ Inferece Settings ]")
    print(f" * Model: {base_model_name}")
    if quantization_method:
        print(f" * Quantizatione: {quantization_method}")
    if use_lora :
        print(" * Low-Lank Adaptation: ", peft_model_dir.split("/")[-1], "\n")

################################################################
## Phrase
data_checker = '''---------- Test Data Check -----------------------------------------------
  Question: {}
  Answer: {}
  Solution: {}
--------------------------------------------------------------------------
'''

## Prompt Engineering ##
prompt_template = {
    "LLAMA2_PROMPT" : "[INST] {} [/INST] ",
    #"YI_PROMPT" : "Human: {} Assistant: ",
    #"SUS_CHAT_PROMPT" : "### Human: {} Assistant: ",
    #"QA_PROMPT" : "[Problem] {} [Solution] "
}
instructions = { # for problem info, which includes 'question' and 'answer'.
    "step-by-step": "주어지는 문제를 단계 별로 풀어 줘. '문제': {} (answer: {})",
    #"role-assign" : "You are a math teacher. Let's solve step-by-step for given for given 'question': {} (answer: {})",
    "problem-info" : "{} (answer: {})"
}
input_template = ""

################################################################
def split_qa_prompt(input_string):
    # regex pattern
    pattern = r"\[Question\]|\[Answer\]|\[Solution\]|\[/INST\]"
    # Split
    split_strings = re.split(pattern, input_string)
    # Remove empty strings from the result
    split_strings = [s.strip() for s in split_strings if s.strip()]
    return split_strings

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

def gen_prompt_dict(ex):
    return {"text": input_template.format(ex["Problem"], ex["Answer"])}

if __name__ == "__main__":
    # Pathes Check
    path_checker(PATHES)
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
    # Use this when select cases
    if test_dataset:
        test_dataset=test_dataset.select(test_idxs)
    
    # Generate prompts
    testset= []
    for prompt in prompt_template.values() :
        for inst in instructions.values() :
            input_template = prompt.format(inst)
            testset.append(test_dataset.map(gen_prompt_dict))
    test_prompts = concatenate_datasets(testset)
    print("test:", test_prompts['text'])
    
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=1024 # defalt = 16 ...
        #top_p=0.5,
    )
    ## About 'temperature' 'and top_p'
    #    'temperature' is a randomness. # under do_sample=True
    #    'top_p' is a minimum probability of next token
    #    'top_k' is a maximum number of token to consider(control not recommended)
    #    Do not change more than two params
    pipe = deploy_llm()
    
    # Inference
    print("#" * 40, "Inference", "#" * 40)
    start_inference_time = time.time()
    
    outputs = pipe.generate(
            KeyDataset(test_prompts, "text"),
            sampling_params,
            lora_request=LoRARequest("Adapter", lora_id, peft_model_dir)
            if lora_id else None )
    
    generated_texts = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print("-"*20, f"<{i+1}'th request>","-"*20)
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: \n{generated_text}")
    end_inference_time = time.time()
    inference_time = end_inference_time - start_inference_time
    print(f"Time Taken for Inference: \n {inference_time:.4f} seconds")