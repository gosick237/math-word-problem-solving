#%%
from transformers import (
    AutoTokenizer,
)
from datasets import load_dataset
import matplotlib.pyplot as plt

# Check List
data_phase = 1
train_dataset_path = f"phase{data_phase}_train.jsonl"
test_dataset_path = f"phase{data_phase}_test.jsonl"

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

def print_dataset_info(dataset):
    print("*"*50)
    print("* Feature: ", dataset.features)
    print("* Train dataset len: ",dataset.__len__())
    print("* Text sample: ",dataset["text"][0])
    print("*"*50)

def gen_prompt_dict(ex):
    return {"text": input_template.format(ex["Problem"], ex["Answer"]) + ex["Solution"] }

def show_seq_len_histogram(dataset):
    #Set tokenizer
    model_dir = "../models/Mixtral-8x7B/Mixtral-8x7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    def generate_and_tokenize_prompt(prompt):
        return tokenizer(gen_prompt(prompt))
    tokenzed_dataset = dataset.map(generate_and_tokenize_prompt)
    plot_data_lengths(tokenzed_dataset['train'])
    plot_data_lengths(tokenzed_dataset['test'])

def gen_prompt(ex):
    return input_template.format(ex["Problem"], ex["Answer"]) + ex["Solution"]

def plot_data_lengths(tokenized_train_dataset):#, tokenized_val_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    #lengths += [len(x['input_ids']) for x in tokenized_val_dataset]
    print(len(lengths))

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    plt.show()

if __name__ == "__main__":

    # Load Dataset
    dataset = load_dataset( 
        "json", 
        data_files={"train":train_dataset_path, "test":test_dataset_path},
        #field= "data",
        #split= "train[30:70]"
    )
    # Generatec Prompt
    process_data = dataset.map(gen_prompt_dict)

    # Max Sequence Analysis
    #show_seq_len_histogram(process_data)

    train_dataset = process_data['train']
    test_dataset = process_data['test']

    # Data Checker
    print_dataset_info(train_dataset)

    # Reduce dataset
    #if train_dataset.__len__() > 1000:
    #    train_dataset = train_dataset.shuffle(seed=41).select(range(1000))
    #    print("split train dataset len: ",train_dataset.__len__())


# %%
