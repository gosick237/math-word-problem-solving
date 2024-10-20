'''
"Description" : This Data Process module is for PRM800K dataset
"Dataset" : {
  "name" : "Wapl Math"
  "shape" : "(16021, 24)"
}
'''
import os
import json
import pandas as pd

dataset_dir = "../../shared_dataset/WAPLMATH/Wapl_math_raw.csv"
target_dir = "."

def save_json(data, target_dir, file_name):
    # File configuration
    new_file_name = os.path.splitext(file_name)[0] + '.jsonl'
    target_path = os.path.join(target_dir, new_file_name)
    # Save
    with open(target_path, 'w') as f:
        for chunk in data:
            json.dump(chunk, f)
            f.write("\n")

def data_key_parser(dataset):
    problem_items=[]
    for k in dataset.keys():
        problem_items.append(k)
        items=[]
        for data in dataset[k]:
            if isinstance(data, str):
                try :
                    for block in json.loads(data):
                        if "type" in block.keys():
                            if not block["type"] in items:
                                items.append(block["type"])
                except :
                    if not type(data) in items :
                        items.append(type(data))
                    pass
            else :
                if not type(data) in items :
                        items.append(type(data))
        print(f"{k}:", items)
        items.clear()

ans_types=['SHORT_ANSWER', 'MULTIPLE_CHOICE', 'TWO_CHOICE']
que_types=['QUESTION_TEXT', 'NOT_PREFACED_SHORT_ANSWER_TEXT', 'PREFACED_MULTIPLE_CHOICE_TEXT', 'QUESTION_IMAGE', 'PREFACED_EXAMPLE_BOX_TEXT', 'NOT_PREFACED_EXAMPLE_BOX_TEXT', 'PREFACED_TWO_CHOICE_TEXT', 'PREFACED_MULTIPLE_CHOICE_IMAGE', 'PREFACED_EXAMPLE_BOX_IMAGE']
sol_types=['SOLUTION_TEXT', 'NOT_PREFACED_SHORT_ANSWER_CORRECT_ANSWER', 'PREFACED_MULTIPLE_CHOICE_CORRECT_ANSWER', 'PREFACED_EXAMPLE_BOX_SOLUTION_TEXT', 'PREFACED_TWO_CHOICE_CORRECT_ANSWER', 'PREFACED_MULTIPLE_CHOICE_SOLUTION_TEXT', 'SOLUTION_IMAGE', 'NOT_PREFACED_EXAMPLE_BOX_SOLUTION_TEXT', 'PREFACED_MULTIPLE_CHOICE_SOLUTION_IMAGE', 'PREFACED_EXAMPLE_BOX_SOLUTION_IMAGE', 'PREFACED_MULTIPLE_CHOICE_TEXT']

class UnkownTypeError(Exception):
    def __init__(self, message):
        self.message = "Unknown TYPE Error \'" + message + "\'"
    
    def __str__(self):
        return self.message

def data_key_parser(dataset):
    problem_items=[]
    for k in dataset.keys():
        problem_items.append(k)
        items=[]
        for data in dataset[k]:
            if isinstance(data, str):
                try :
                    for block in json.loads(data):
                        if "type" in block.keys():
                            if not block["type"] in items:
                                items.append(block["type"])
                except :
                    if not type(data) in items :
                        items.append(type(data))
                    pass
            else :
                if not type(data) in items :
                        items.append(type(data))
        print(f"{k}:", items)
        items.clear()

# Example usage:
if __name__ == "__main__":
    # process on a file
    dataset = pd.read_csv(dataset_dir, header=0)
    print("Orginal Dataset Size: ", dataset.shape)
    dataset = pd.DataFrame(dataset)[dataset['STATUS']=='ACCEPT']
    print("Accepted Dataset Size: ", dataset.shape)

    # Key Parsing
    data_key_parser(dataset)
    
    # Process
    new_dataset = [] # jsonl
    numsss=[]
    cnt_unused_prob=0
    for idx, row in dataset.iterrows():
        question = ""
        solution = ""
        answer = []
        
        useImg = False
        checklist= []
        temp=0
        
        # Build Question
        for block in json.loads(row['QUESTION']):
            #print(block)
            if block['type'] == 'QUESTION_TEXT':
                for l in block['data']:
                    question += l
            elif 'CHOICE_TEXT' in block['type'] :
                question += "\n"
                for idx, l in enumerate(block['data']):
                    question += f"{idx+1}. {l}  "
            elif 'BOX_TEXT' in block['type']:
                question += "|"
                for l in block['data']:
                    question += f" {l}"
                question += "| "
            elif 'IMAGE' in block['type'] : #and 'PREFACED' in block['type'] :
                # Exclude Problems that include Example or Choice with images
                useImg = True
            else :
                cnt_unused_prob += 1
                if not block['type'] in checklist:
                    checklist.append(block['type'])
        if useImg:
            continue
        
        # Build Solution & Answer
        for block in json.loads(row['SOLUTION']):
            if 'type' in block.keys():
                # build sol
                if 'SOLUTION_TEXT' in block['type']:
                    for l in block['data']:
                        solution += l
                # build ans
                ans_type= row['ANSWER_TYPE']
                if ans_type == 'SHORT_ANSWER':
                    if block['type'] == 'NOT_PREFACED_SHORT_ANSWER_CORRECT_ANSWER' :
                        for l in block['data']:
                            answer.append(l) # String answer
                elif ans_type == 'MULTIPLE_CHOICE' or ans_type == 'TWO_CHOICE' :
                    if 'CHOICE_CORRECT' in block['type'] :
                        for l in block['data']:
                            answer.append(l+1) # Int answer
            else : 
                # Data type is missing and Answer type is 'MULTIPLE_CHOICE'
                for l in block['data']:
                    solution += " ∙ " + l + "\n"
        # Special Case
        if len(question) > 1 and len(answer) <1:
            answer.append(2)
        
        if temp:
            #print("ID:", row['STATUS'])
            print("-"*50)
            print("ID:", row['PROB_ID'])
            print("Answer Type: ", row['ANSWER_TYPE'])
            print("origin_Question: ", row['QUESTION'])
            print("origin_Solution: ", row['SOLUTION'])
            print("--- new ---")
            print("Question: ", question)
            print("Solution: ", solution)
            print("Answer:", answer)
            print("")
        new_dataset.append({
            "QUESTION" : question,
            "SOLUTION" : solution,
            "ANSWER" : answer
        })
    print(" * num of new data: ", len(new_dataset))
    print(" * new data samples:\n", new_dataset[:5])

    print("what is left: ",checklist)
    print("how many problem excluded: ",cnt_unused_prob)