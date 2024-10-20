import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    logging
)
from peft import PeftModel
from datasets import load_dataset
import re
import time
import os

## Model & Data Path ##
# Base Model
model_dir = "./models/Mixtral-Instruct/"
base_model_name = "Mixtral-8x7B-Instruct-v0.1"
base_model_dir = model_dir + base_model_name
# Adapter
dataset_name = "PRM800K"
data_phase = 2
checkpoint = "900"
peft_model_dir = model_dir + f"Adapter-{dataset_name}_phase{data_phase}-cp{checkpoint}"

# Testset
testset_path = "data/phase1_test.jsonl"
#test_idxs = [0]
test_idxs = [0,5,22]

PATHES = [testset_path, base_model_dir, peft_model_dir]

## QLoRA Quantization Parameters ##
use_4bit = True # Activate 4-bit precision base model loading
bnb_4bit_compute_dtype = torch.bfloat16 #"float16"
bnb_4bit_quant_type = "nf4" # fp4 or nf4
use_nested_quant = True # double quantization

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

    # Quantization Configurations
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Set base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        quantization_config=bnb_config,
        device_map=device_map,
        attn_implementation="flash_attention_2", # for Inference Efficiency
        low_cpu_mem_usage=True,
        #return_dict=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    model = PeftModel.from_pretrained(base_model, peft_model_dir)
    #model = model.merge_and_unload()

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
        #temperature = 0.2,
        max_length=1024
    )
    ## About 'temperature' 'and top_p'
    #    'temperature' is a randomness.
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

if __name__ == "__main__":
    # Pathes Check
    path_checker(PATHES)

    # testset
    test_dataset = load_dataset(
        "json",
        data_files={"test":testset_path},
        split='test'
    )
    test_data = []

    # set llm
    pipe = deploy_llm()
    
    sit = time.time()
    # Inferece per Test
    for i, id in enumerate(test_idxs):
        print("#" * 10, f"{i+1}'th Test", "#" * 50)
        prob = test_dataset[id]['Problem']
        ans = test_dataset[id]['Answer']
        sol = test_dataset[id]['Solution']
        print (data_checker.format(prob,ans,sol))

        # inference per prompt template
        for p_key, p_val in prompt_template.items():
            for i_key, i_val in instructions.items():
                given_info = p_val.format(i_val).format(prob,ans)
                print(f"['{p_key}'-style prompt template with '{i_key}' instruction]\n  ", given_info)
            
                # run inference
                start_inference_time = time.time()
                result = pipe(given_info)
                end_inference_time = time.time()
                inference_time = end_inference_time - start_inference_time
                print(f"Time Taken for Inference: \n {inference_time:.4f} seconds")

                # check response
                output_split = split_qa_prompt(result[0]['generated_text'])
                print("########## Inference ##########")
                print("[Generator]", output_split[-1], "\n")
                print("#" * 50)
    eit = time.time()
    inf_time = eit - sit
    print(f"Total Time Taken for Inference: \n {inf_time:.4f} seconds")
