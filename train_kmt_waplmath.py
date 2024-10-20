import torch
import torch.utils.tensorboard
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import time, random
import nltk.translate.bleu_score as bleu
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    GenerationConfig
)
from utils import (
    init_gpu,
    show_gpu_utilization,
    show_dataset_info,
    path_checker,
    vllm_lora_process,
    preprocess_logits_for_metrics
)

# 'bnb' could be deprecated

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

## Model & Data Path ##
# Base Model
base_model = "Komistral-Instruct"
base_model_name = "Komistral-7B-Instruct"
base_model_dir = f"../shared_model/{base_model}/{base_model_name}"
# Adapter
dataset_name = "WAPLMATH"
peft_model_dir = f"models/{base_model}/Adapter-{dataset_name}"

# Dataset Path
train_dataset_path = f"data/{dataset_name}/wapl_math_7471.jsonl"
val_rate = 0.1

PATHES = [base_model_dir, train_dataset_path]

###############################
## Quantization (bnb) ##
quantization_method = None # 'bnb' | None | 'awq' | 'squeezellm'

if quantization_method:
    base_model_dir = base_model_dir + "-" + quantization_method

###############################
## LoRA Parameters ##
lora_r = 32  # LoRA attention dimension
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
per_device_train_batch_size = 8
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1 # increas for small batch size
gradient_checkpointing = True # activate for lower memory in weight saving on back-propagation
max_grad_norm = 0.3 # Maximum gradient normal (gradient clipping)
learning_rate = 1e-5 # Initial learning rate (AdamW optimizer)
weight_decay = 0.001 # applied to all layers except bias/LayerNorm weights
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1 # Number of training steps (overrides num_train_epochs)
warmup_ratio = 0.03 # Ratio of steps for a linear warmup (from 0 to learning rate)
group_by_length = True # Group sequences into batches with same length (# Saves memory and speeds up training considerably)
save_steps = 100 # Save checkpoint every X updates steps
logging_steps = 50 # Log every X updates steps
eval_steps = 100

## Prompt Engineering ##
prompt_template = {
    "LLAMA2_PROMPT" : "[INST] {} [/INST] ",
    "YI_PROMPT" : "Human: {} Assistant: ",
    "SUS_CHAT_PROMPT" : "### Human: {} Assistant: ",
    "QA_PROMPT" : "[Problem] {} [Solution] "
}
instructions = { # for problem info, which includes 'question' and 'answer'.
    "step-by-step": "주어진 문제에 대해 단계 별로 풀어줘. '문제': {} (답: {})",
    "role-assign" : "너는 수학 선생이야, 주어지는 문제에 대해 단계 별로 설명해줘 '문제': {} (답: {})",
    "problem-info" : "{} (답: {})"
}
input_template = prompt_template["LLAMA2_PROMPT"].format(instructions['problem-info'])

################################################################################
# Functions

###############################
## Training parameters ##
def gen_prompt_dict(ex):
    return {"text": input_template.format(ex["Problem"], ex["Answer"]) + ex["Solution"] }

def set_trainer(train_dataset, val_dataset):#, test_dataset):
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
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    #Set tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_dir, 
        padding_side = "right", # Fix weird overflow issue with fp16 training
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
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

    generation_arguments = GenerationConfig(
        do_sample=True,
        top_k=50,
        #num_beams=2
    )
    #generation_arguments.save_pretrained("/tmp", "custom_generation_config.json")

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
        report_to="tensorboard",
        #generation_arguments=generation_arguments
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
            print("Label : ", decoded_labels[idx])
            print("Preds : ", decoded_preds[idx].strip())
            print("="*60)

        #Custom
        acc = 0.0
        bleu_s = 0.0
        for gt, pred in zip(decoded_labels, decoded_preds):
            if pred.strip() == gt.strip():
                acc += 1.0
            # bleu score 4-gram
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

        return {
            "acc": round(acc/len(decoded_labels), 4),
            "bleu4": round(bleu_s/len(decoded_labels), 4)
        }

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        #peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        dataset_text_field="text",  # this is the text column in dataset 
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )
    print("memory After set Trainer:")
    show_gpu_utilization()

    end_loading_time = time.time()
    loading_time = end_loading_time - start_loading_time
    print(f"Time Taken for Load: \n {loading_time:.4f} seconds")

    return trainer

if __name__ == "__main__":
    # Pathes Check
    path_checker(PATHES)
    # Clear GPU
    init_gpu()

    ###### Dataset #####
    print("*"*30, " Data Check ", "*"*30)
    dataset = load_dataset( 
        "json", 
        data_files={"train":train_dataset_path},
    )
    process_data = dataset.map(gen_prompt_dict)
    process_data = process_data['train'].train_test_split(test_size=val_rate)
    train_dataset = process_data['train'].shuffle(seed=42)
    val_dataset = process_data['test']
    show_dataset_info(train_dataset, "Check Train Dataset")
    show_dataset_info(val_dataset, "Check Train Dataset")

    ##### Training #####
    trainer = set_trainer(train_dataset, val_dataset)#, test_dataset)
    print("\n", "*"*100)
    print("<<<<< On Learning >>>>")
    trainer.train()

    # Save trained model
    trainer.model.save_pretrained(peft_model_dir, save_embedding_layers=False)
    #trainer.tokenizer.save_pretrained(peft_model_dir)
    print(f"{peft_model_dir} is saved")

    # Eval
    print(trainer.evaluate())

    print("++ Remove Embedding Layer (vllm support adapter only) ++")
    vllm_lora_process(output_dir)