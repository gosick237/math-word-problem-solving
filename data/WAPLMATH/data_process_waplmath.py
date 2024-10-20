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

dataset_dir = "../../../shared_dataset/WAPLMATH/wapl_math.csv"
target_dir = "."

def save_json(data, target_dir, file_name):
    # File configuration
    new_file_name = os.path.splitext(file_name)[0] + '.jsonl'
    target_path = os.path.join(target_dir, new_file_name)
    # Save
    with open(target_path, 'w', encoding="utf-8") as f:
        for chunk in data:
            json.dump(chunk, f, ensure_ascii=False)
            f.write("\n")
    print(f"'{target_path}' has been saved")

ans_types=['SHORT_ANSWER', 'MULTIPLE_CHOICE', 'TWO_CHOICE']
que_types=['QUESTION_TEXT', 'NOT_PREFACED_SHORT_ANSWER_TEXT', 'PREFACED_MULTIPLE_CHOICE_TEXT', 'QUESTION_IMAGE', 'PREFACED_EXAMPLE_BOX_TEXT', 'NOT_PREFACED_EXAMPLE_BOX_TEXT', 'PREFACED_TWO_CHOICE_TEXT', 'PREFACED_MULTIPLE_CHOICE_IMAGE', 'PREFACED_EXAMPLE_BOX_IMAGE']
sol_types=['SOLUTION_TEXT', 'NOT_PREFACED_SHORT_ANSWER_CORRECT_ANSWER', 'PREFACED_MULTIPLE_CHOICE_CORRECT_ANSWER', 'PREFACED_EXAMPLE_BOX_SOLUTION_TEXT', 'PREFACED_TWO_CHOICE_CORRECT_ANSWER', 'PREFACED_MULTIPLE_CHOICE_SOLUTION_TEXT', 'SOLUTION_IMAGE', 'NOT_PREFACED_EXAMPLE_BOX_SOLUTION_TEXT', 'PREFACED_MULTIPLE_CHOICE_SOLUTION_IMAGE', 'PREFACED_EXAMPLE_BOX_SOLUTION_IMAGE', 'PREFACED_MULTIPLE_CHOICE_TEXT']
status_enum=['DELETED', 'ACCEPT', 'DRAFT', 'REJECT']

# Example usage:
if __name__ == "__main__":
    # process on a file
    dataset = pd.read_csv(dataset_dir, header=0)
    print("Orginal Dataset Size: ", dataset.shape)
    dataset = pd.DataFrame(dataset)[dataset['STATUS']=='ACCEPT']
    print("Accepted Dataset Size: ", dataset.shape)
    
    # Process
    new_dataset = [] # jsonl
    checklist= []
    for idx, row in dataset.iterrows():
        question = ""
        solution = ""
        answer = []

        useImg = False
        
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
                question += "| "
                for l in block['data']:
                    question += f" {l}"
                question += "  |"
            elif 'IMAGE' in block['type'] and 'PREFACED' in block['type'] :
                # Exclude Problems that include Example or Choice with images
                useImg = True # break
            else :
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
                            answer.append(str(l+1)) # Int answer to String
            else : 
                # Data type is missing and Answer type is 'MULTIPLE_CHOICE'
                for l in block['data']:
                    solution += " ∙ " + l + "\n"
        # Special Case
        if len(question) > 1 and len(answer) <1:
            answer.append("2")
        
        new_dataset.append({
            "Problem" : question,
            "Solution" : solution,
            "Answer" : ', '.join(answer) if len(answer) else ""
        })
    
    print("########## New Dataset ##########")
    save_json(new_dataset, target_dir, os.path.basename(dataset_dir))
    print(" * num of new data: ", len(new_dataset))
    print(" * new data samples:\n", new_dataset[:5])

    print("unused types:", checklist)