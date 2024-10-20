import torch
import torch.utils.tensorboard
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb
import pandas as pd
import time, random
import nltk.translate.bleu_score as bleu
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from utils import (
    init_gpu,
    show_gpu_utilization,
    show_dataset_info,
    path_checker,
    vllm_lora_process,
    preprocess_logits_for_metrics,
    call_prompt_template,
    remove_repeated_sequences,
    split_with_pattern,
    get_completion_with_ans
)

################################################################################
'''
## Note! here For Exprement ##
[CheckList before run 'train_mwps.py']
    - Prerequisite
        - Model :
            Put 'base_model' folder in 'models' folder:
            > Refer to 'model_loader.py'
            # base model == foundation model
        - Dataset :
            Put files in 'data' folder
            > Refer to 'data_process.py'
    - Parameters
        - Pathes
            "model_dir" : base model path
            "peft_model_dir" : finetuned model path (== Adapter)
            "train_dataset_path" : train data path
            "test_dataset_path" : test data path !! If you don't need, change this code !!
        - Training Options
            "batch_size" : 4NF-quatized 7B model takes 1.5 ~ 2GB memories per a batch
            "
'''
################################################################################
# Parameters
###############################

###############################
## Path ##

# Base Model
base_model = "Llama3-8B-Instruct"
base_model_name = "Llama-3-8B-Instruct"

# Quantization
quantization_method = 'awq' # 'bnb' | None | 'awq' | 'squeezellm'
if quantization_method :
    base_model += f"-{quantization_method}"
    base_model_name += f"-{quantization_method}"
shared_model_dir = f"../shared_model/{base_model}"
base_model_dir = f"{shared_model_dir}/{base_model_name}"

# Adapter
dataset_name = "PRM800K"
peft_model_dir = f"{shared_model_dir}/Adapter-{dataset_name}"

# Dataset Path
train_dataset_path = f"data/{dataset_name}/train.jsonl"
train_data_limit = None #10000
val_rate = 0.02

PATHES = [base_model_dir, train_dataset_path]
path_checker(PATHES)

###############################
## Logging: Wandb ##
project_name = "psa-mwps"

###############################
## Prompt ##
prompt_style = "llama3"
instruction_style = "problem_answer"
input_template, prompt_template= call_prompt_template(prompt_style, instruction_style)
INSTRUCTION_TEMPLATE=prompt_template[0]
RESPONSE_TEMPLATE=prompt_template[-1]

###############################
## LoRA Parameters ##
lora_r = 16  # LoRA attention dimension
lora_alpha = 16 # LoRA scaling (kind of learning rate)
lora_dropout = 0.1

###############################
## SFT parameters ##
max_seq_length = 1500 #None # default 1024
packing = False # Pack multiple short examples in the same input sequence to increase efficiency
device_map = "auto"
# device_map = {"": 0} # for Load the entire model on the GPU 0

###############################
## Training parameters ##
output_dir = "./results"
num_train_epochs = 1
# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = True #False
per_device_train_batch_size = 4
per_device_eval_batch_size = 2
gradient_accumulation_steps = 2 # increas for small batch size
gradient_checkpointing = True # activate for lower memory in weight saving on back-propagation
max_grad_norm = 0.3 # Maximum gradient normal (gradient clipping)
learning_rate = 1e-5 # Initial learning rate (AdamW optimizer)
weight_decay = 0.001 # applied to all layers except bias/LayerNorm weights
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1 # Number of training steps (overrides num_train_epochs)
warmup_ratio = 0.03 # Ratio of steps for a linear warmup (from 0 to learning rate)
group_by_length = True # Group sequences into batches with same length (# Saves memory and speeds up training considerably)
label_smoothing_factor=0.05
save_steps = 100 # Save checkpoint every X updates steps
logging_steps = 50 # Log every X updates steps
eval_steps = 100

################################################################################
# Functions

###############################
## Data Processing ##
def gen_prompt(example):
    # For Training Prompt
    # Prompt == instruction + completion; open_QA-type)
    output_texts = []
    for i in range(len(example['solution'])):
        text = input_template.format(example["problem"][i], example["answer"][i]) + "\n\n" + example["solution"][i] + "<|end_of_text|>"
        output_texts.append(text)
    return output_texts

if __name__ == "__main__":
    # Clear GPU
    init_gpu()

    ####################
    ##### Dataset Setting #####    
    run_pp = wandb.init( project=project_name, name="prompt_llama3", job_type="data_preprocess")
    artifact = run_pp.use_artifact('gosick237_psa/psa-mwps/prm800k_llama3:v0', type='dataset')
    artifact_dir = artifact.download() # download at os.getcwd() + 'artifacts/{Project_name}/'

    dataset = load_dataset( "json", data_dir=artifact_dir, split='train' )
    #show_dataset_info(dataset, "Check Original Dataset")
    if train_data_limit:
        dataset = dataset.select(range(train_data_limit))
    #dataset = dataset.map(lambda ex: ex["solution"] += "<|end_of_text|>")

    # Split
    dataset = dataset.train_test_split(test_size=val_rate)
    train_dataset = dataset['train'].shuffle(seed=42)
    val_dataset = dataset['test']
    show_dataset_info(train_dataset, "Check Train Dataset")
    show_dataset_info(val_dataset, "Check Train Dataset")

    # Log# Log
    art_train = wandb.Artifact( name="PRM800k_train", type="dataset_split" )
    art_valid = wandb.Artifact( name="PRM800k_valid", type="dataset_split" )

    train_table = wandb.Table(dataframe=pd.DataFrame(train_dataset))
    val_table  = wandb.Table(dataframe=pd.DataFrame(val_dataset))
    run_pp.log({"train_dataset":train_table, "val_dataset":val_table})

    art_train.add(train_table, "PRM800k_train")
    art_valid.add(val_table, "PRM800k_valid")
    run_pp.log_artifact(art_train)
    run_pp.log_artifact(art_valid)

    run_pp.finish()

    ####################
    ##### Training ##### 
    start_loading_time = time.time()

    # Set base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        device_map=device_map,
        attn_implementation="flash_attention_2", # for Inference Efficiency
        torch_dtype=torch.float16, # for flash_attn && float16 for AWQ efficiency
        local_files_only=True,
        **{"low_cpu_mem_usage": True, "use_cache": False}
    )
    model.enable_input_require_grads() # when train without LM_head
    #if torch.cuda.device_count() > 1:
    #    model.is_parallelizable = True
    #    model.model_parallel = True

    #Set tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_dir, 
        padding_side = "right", # Fix weird overflow issue with fp16 training
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.add_special_tokens({"pad_token":"<pad>"}) # llama3 cannot use -1. & eos could incur to endless text
    model.config.pad_token_id = tokenizer.pad_token_id
    
    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        instruction_template=INSTRUCTION_TEMPLATE,
        tokenizer=tokenizer
    ) # Don't use "\n\n" in thoses template
    
    # Check labels (ft. collator)
    #sample = gen_prompt(val_dataset[0])
    #collator_output = collator([tokenizer(sample)])
    #print("sample: ", sample)
    #print("collator_out: ", collator_output)

    print("memory After set Model:")
    show_gpu_utilization()

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        # for mistral.
        target_modules=[
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj", 
            "gate_proj",
            "up_proj",
            "down_proj",
            #"lm_head"
        ]
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    run_train = wandb.init(
        project=project_name,
        name="train_Llama3_8B_instruct",
        job_type="train",
        tags=["Llama-3-8B", "Instuct", "SFT_trainer_lora"]
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=save_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant':True}, #이게 False면 16-bs도 안 돌아감
        optim=optim,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        label_smoothing_factor=label_smoothing_factor,
        report_to="wandb"
    )

    # Validation
    def compute_metrics (pred):
        # labels_ids = pred.label_ids
        # pred_ids = pred.predictions
        pred_ids, labels_ids = pred

        # In pytorch -100 corresponds to things should be ignored
        # Trainer() use -100 for padding, hence, we gotta change it for decoding.
        pred_ids[pred_ids == -100] = tokenizer.pad_token_id
        #pred_ids= np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True) #clean_up_tokenization_spaces=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        decoded_labels = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        
        # Show Samples
        result_sample_size = 3
        show_idx = random.randint(0 , len(decoded_labels) - result_sample_size) 
        for idx in range(show_idx, show_idx + result_sample_size):
            print("="*20, "Validation Samples", "="*20)
            print("Label : ", decoded_labels[idx].strip())
            print("-"*20)
            print("Preds (Origin): ", decoded_preds[idx].strip())

            pred_split = split_with_pattern(r"assistant\n\n", decoded_preds[idx])
            inst = pred_split[0]
            compl = pred_split[-1]
            compl, ans = get_completion_with_ans(compl)
            print(f"Preds (Completion) : ", compl)
            print(f"Preds (Answer) : ", ans)
            print("="*60)
        
        #Custom
        num_data_without_ans = 0
        num_eval = len(decoded_labels)
        acc = 0.0
        bleu_s = 0.0
        for gt, pred in zip(decoded_labels, decoded_preds):

            # Completion Extraction
            # instruction = split_with_pattern(r"assistant\n\n", pred)[0]
            completion = split_with_pattern(r"assistant\n\n", pred)[-1]

            # Accuracy
            try:
                corr_ans = ''
                if "# Answer" in gt:
                    corr_ans = gt.split("# Answer")[1].strip()
                else:
                    last_line = split_with_pattern(r"\n", gt)[-1]
                    corr_ans = last_line.split('  ')[-1].strip()
                    num_data_without_ans =+ 1
                pred, pred_ans = get_completion_with_ans(completion)

                if corr_ans == pred_ans :
                    acc += 1.0
            except:
                print("Cannot extract answer")
                print("---gt: ", gt)
                print("---pred: ", pred)
            
            # BLEU score 4-gram
            try:
                bleu_score = bleu.sentence_bleu(
                    list(map(lambda ref: ref.split(), [gt])),
                    pred.split(),
                    auto_reweigh=True,
                    smoothing_function=bleu.SmoothingFunction().method2
                )
            except Exception as e:
                bleu_score = 0.0
            bleu_s += bleu_score
        
        print("\n\n", f" ++ Number of pred with answers: {num_eval - num_data_without_ans}/{num_eval}")

        return {
            "acc": round(acc/num_eval, 4),
            "bleu4": round(bleu_s/num_eval, 4)
        }

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        #tokenizer=tokenizer
        #peft_config=peft_config,
        max_seq_length=max_seq_length,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        #dataset_text_field="text",  # this is the text column in dataset 
        formatting_func=gen_prompt,
        data_collator=collator,
        #packing=packing, # ConstantLengthDataset(..., seq_length=1024) && not supported with collator
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        args=training_arguments,
    )
    print("memory After set Trainer:")
    show_gpu_utilization()

    end_loading_time = time.time()
    loading_time = end_loading_time - start_loading_time
    print(f"Time Taken for Load: \n {loading_time:.4f} seconds")
    
    print("\n", "*"*100)
    print("<<<<< On Learning >>>>")
    trainer.train()

    # Save trained model
    trainer.model.save_pretrained(peft_model_dir, save_embedding_layers=False)
    trainer.tokenizer.save_pretrained(peft_model_dir)
    #trainer.tokenizer.save_pretrained(peft_model_dir)
    print(f"{peft_model_dir} is saved")

    # Eval
    print(trainer.evaluate())

    print("++ Remove Embedding Layer (vllm support adapter only) ++")
    vllm_lora_process(output_dir)

    run_train.finish()
