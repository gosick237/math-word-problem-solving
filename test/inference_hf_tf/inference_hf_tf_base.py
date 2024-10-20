import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    logging
)
from datasets import load_dataset
import re
import time

## Model data Parameters ##
model_dir = "./models/Mistral-7B-Instruct-v0.2"

testset_path = "data/phase1_test.json"
test_num = [0, 5]

## Phrase
data_checker = "########## Test Data Check ##########\n  Question: {}\n  Answer: {}\n  Solution: {}\n####################################"
prompt_template = {
    "step-by-step" : """Let's solve step by step. [Question]{} [Answer]{}""",
    "role-instuct" : """You're a math teacher. With given math word problem info, solve step by step. [Question]{} [Answer]{}""",
    "question-solution" : """[Question]{} [Answer]{} [Solution]""",
    "chat-template" : """[INST] Let's solve step by step for given [Question]{} [Answer]{} [/INST]"""
}

def split_prompt(input_string):
    # regex pattern
    pattern = r"\[Question\]|\[Answer\]|\[Solution\]"
    # Split
    split_strings = re.split(pattern, input_string)
    # Remove empty strings from the result
    split_strings = [s.strip() for s in split_strings if s.strip()]
    return split_strings

def deploy_llm():
    # Set Model
    start_loading_time = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"#{"": 0},
    )
    model = base_model

    # Set Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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

if __name__ == "__main__":
    # testset load
    dataset = load_dataset(
        "json",
        data_files={"test":testset_path},
        field= "data",
        split= "test"
    )

    # set llm
    pipe = deploy_llm()

    
    # Inferece per Test
    for i, test in enumerate(test_num):
        print("#" * 20, f"{i}'th Test", "#" * 20)
        str = split_prompt(dataset[test_num]['text'])
        print (data_checker.format(str[0],str[1],str[2]))

        # inference per prompt
        for key, val in prompt_template.items():
            # set prompt
            given_info = val.format(str[0],str[1])
            print(f"[{key}style prompt_template]\n  ", given_info)
            
            # run inference
            start_inference_time = time.time()
            result = pipe(given_info)
            end_inference_time = time.time()
            inference_time = end_inference_time - start_inference_time
            #print("\nresult:\n  ", result[0]['generated_text'], "\n")
            print(f"Time Taken for Inference: \n {inference_time:.4f} seconds")

            # check response
            output_split = split_prompt(result[0]['generated_text'])
            print("########## Inference ##########")
            print("[Generator]", output_split[-1], "\n")
            print("#" * 50)