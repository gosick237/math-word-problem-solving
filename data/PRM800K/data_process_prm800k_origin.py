'''
"Description" : This Data Process module is for PRM800K dataset
"Dataset" : {
  "name" : "PRM800K"
  "url" : "https://github.com/openai/prm800k/tree/main"
}
'''
import os
import json
import numpy as np

# path
dataset_dir = "../../shared_dataset/PRM800K/prm800k/data/"
target_dir = "."
type_error_cnt = 0

def gen_qa(line):
    item = json.loads(line)
    if item['label']['finish_reason'] == 'solution':
        return {
            "problem": item['question']['problem'],
            "answer" : item['question']['ground_truth_answer'],
            "solution" : gen_solution(item['label']['steps'])
        }
    else :
        return None

def gen_solution(steps):
    # Generate step-by-step solution by concatenating step.
    # print("  Solution: ")
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

            step_by_step_solution += completion_text + "\n"
            
    return step_by_step_solution.strip()

def save_json(data, target_dir, file_name):
    # File configuration
    new_file_name = os.path.splitext(file_name)[0] + '.jsonl'
    target_path = os.path.join(target_dir, new_file_name)
    # Save
    with open(target_path, 'w', encoding="utf-8") as f:
        for chunk in data:
            json.dump(chunk, f)
            f.write("\n")

# Example usage:
if __name__ == "__main__":
    # process on each file
    for file in os.listdir(dataset_dir):
        if os.path.isfile(os.path.join(dataset_dir, file)):
            # process
            process_data = []
            with open(os.path.join(dataset_dir, file), 'r') as f:
                for line in f:
                    solved_data = gen_qa(line)
                    if solved_data:
                        process_data.append(solved_data)
            # save
            save_json(process_data, target_dir, os.path.basename(file))