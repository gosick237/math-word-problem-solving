import os
import numpy as np
import json
from datasets import load_dataset

# path
dataset_dir = "../../Datasets/PRM800K/prm800k/data/"
target_dir = "."
# path
dataset_path = "../../Datasets/PRM800K/prm800k/data/phase1_test.jsonl"

## Phrase
data_checker = "########## Test Data Check ##########\n  Question: {}\n  Answer: {}\n###############################"

## Prompt Engineering ##
prompt_template = {
    "LLAMA2_PROMPT" : "[INST] {} [/INST] ",
    "YI_PROMPT" : "Human: {} Assistant: ",
    "SUS_CHAT_PROMPT" : "### Human: {} Assistant: ",
    "QA_PROMPT" : "[Problem] {} [Solution] "
}
instructions = { # for problem info, which includes 'question' and 'answer'.
    "step-by-step": "Let's solve step-by-step for given 'question' is '{}' and 'answer' is '{}'",
    "role-assign" : "You are a math teacher. Let's solve step-by-step for given for given 'question' is '{}' and 'answer' is '{}'",
    "problem-info" : "{} (answer: {})"
}
input_template = prompt_template["QA_PROMPT"].format(instructions['step-by-step'])

def gen_solution(steps):
    # Generate step-by-step solution by concatenating step.
    #print("  Solution: ")
    step_by_step_solution = ""
    for i, step in enumerate(steps):
        if step['completions'] :
            # Select Best Step
            #   1st opt: "chosen_completion"
            #   2nd opt: "human_completion"
            #       this key could be null, int, str (data error...)
            #   3rd opt: null
            #       chose random step in rated over 0. (;rate : [-1,0,1])
            if step['completions'].__len__() > 1:
                if step["chosen_completion"] :
                    completion_text = step['completions'][step["chosen_completion"]]['text']
                else :
                    if step["human_completion"]:
                        if isinstance(step["human_completion"], int):
                            completion_text = step['completions'][step["human_completion"]]['text']
                        else :
                            completion_text = step["human_completion"]['text']
                    else:
                        # Filtering if 'rating(score)' is less than 0
                        alternatives = list(filter(lambda x: x["rating"] != None and x["rating"] >= 0 ,step['completions']))
                        if alternatives.__len__() > 1:
                            # random pick if num of alternatives is over 2
                            seed = np.random.randint(0, alternatives.__len__())
                            completion_text = alternatives[seed]['text']
                        else:
                            # skip if there's no appropriate step.
                            completion_text = ''
            else:
                # Most Step are single-ton
                completion_text = step['completions'][0]['text']

            # Cleaning : Most last step include redundant '\n'
            if i == len(steps) - 1:
                completion_text = completion_text.replace('\n', ' ')

            # Check & Merge
            print(f"    [step_{i}]: ", completion_text)
            if step['completions'].__len__() > 1:
                for j, completion in enumerate(step['completions']):
                    print(f"\tAlt_{j}: ", completion)
            step_by_step_solution += completion_text + "\n"
            
    return step_by_step_solution.strip()

if __name__ == "__main__":
    
    # process on each file
    qa_data = []
    target = 23
    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            item = json.loads(line)

            if i == target :
                if item['label']['finish_reason'] == 'solution':
                    question_text = item['question']['problem']
                    answer_text = item['question']['ground_truth_answer']
                    steps = item['label']['steps']
                    
                    print(data_checker.format(question_text, answer_text))
                    print(f"{target} finished reason:", item['label']['finish_reason'])                
                    solution_text = gen_solution(steps)
                    input_text = input_template.format(question_text, answer_text) + solution_text
                    qa_data.append({"text": input_text})
                    print("****** qa_data example *****\n  ", input_text)
                    break
                else :
                    target=i+1