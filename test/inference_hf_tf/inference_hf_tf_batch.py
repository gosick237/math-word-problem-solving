import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    logging
)
from transformers.pipelines.pt_utils import KeyDataset
from peft import PeftModel
from datasets import load_dataset, concatenate_datasets
import re
import time
import os

## Model & Data Path ##
# Base Model
base_model = "Mistral-Instruct"
base_model_name = "Mistral-7B-Instruct-v0.2"
base_model_dir = f"../shared_model/{base_model}/{base_model_name}"
# Adapter
dataset_name = "PRM800K"
data_phase = 2
checkpoint = "300"
peft_model_dir = f"models/{base_model}/Adapter-{dataset_name}_phase{data_phase}-cp{checkpoint}"

# Testset
testset_path = "data/phase1_test.jsonl"
test_idxs = [0,5,22]

inf_batch_size = 6

PATHES = [testset_path, base_model_dir, peft_model_dir]

## Resource Pamameters ##
device_map = "auto" # {"": 0}

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
    "step-by-step": "Let's solve step-by-step for given 'question': {} (answer: {})",
    #"role-assign" : "You are a math teacher. Let's solve step-by-step for given for given 'question': {} (answer: {})",
    "problem-info" : "{} (answer: {})"
}
input_template = ""

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

    # Set base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        device_map=device_map,
        attn_implementation="flash_attention_2", # for Inference Efficiency
        low_cpu_mem_usage=True,
        #return_dict=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    model = PeftModel.from_pretrained(base_model, peft_model_dir)
    model = model.merge_and_unload()

    # Set Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.padding_side = "right"

    # Ignore warnings
    logging.set_verbosity(logging.CRITICAL)

    # Run text generation pipeline with our next model
    pipe = pipeline(
        task="text-generation", 
        model=model,
        tokenizer=tokenizer,
        #top_p = 0.9
        #do_sample=True,
        #temperature = 0.4,
        max_length=1024
    )
    ## About 'temperature' 'and top_p'
    #    'temperature' is a randomness. # under do_sample=True
    #    'top_p' is a minimum probability of next token
    #    'top_k' is a maximum number of token to consider(control not recommended)
    #    Do not change more than two params
    end_loading_time = time.time()
    loading_time = end_loading_time - start_loading_time
    print(f"Time Taken for Load: \n {loading_time:.4f} seconds")
    return pipe

def path_checker(pathes):
    for path in pathes:
        if not os.path.exists(path):
            print(f"Given path '{path}' is not exist. Check your path!")
            quit()

def gen_prompt_dict(ex):
    return {"text": input_template.format(ex["Problem"], ex["Answer"])}

if __name__ == "__main__":
    # Pathes Check
    path_checker(PATHES)

    # testset
    test_dataset = load_dataset(
        "json",
        data_files={"test":testset_path},
        split='test'
    )
    test_dataset=test_dataset.select(test_idxs)
    
    # Generate prompts
    testset= []
    for prompt in prompt_template.values() :
        for inst in instructions.values() :
            input_template = prompt.format(inst)
            testset.append(test_dataset.map(gen_prompt_dict))
    test_prompts = concatenate_datasets(testset)
    #print("test:", test_prompts['text'])

    # set llm
    pipe = deploy_llm()
    
    # Inference
    print("#" * 40, "Inference", "#" * 40)
    start_inference_time = time.time()
    for idx, out in enumerate(pipe(
            KeyDataset(test_prompts, "text"),
            batch_size=inf_batch_size,
            truncation="only_first" )):
        print("[Prompt]: \n",test_prompts[idx]['text'])
        print("[Generator]: \n", split_qa_prompt(out[0]['generated_text'])[-1], "\n")
        print("#" * 50)
    end_inference_time = time.time()
    inference_time = end_inference_time - start_inference_time
    print(f"Time Taken for Inference: \n {inference_time:.4f} seconds")