import os
import logging

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from pynvml import *

os.environ["WANDB_PROJECT"] = "nous-finetunes"

BASE_MODEL_NAME = "models/nous-six-base"


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


# Model setup
tokenizer = LlamaTokenizer.from_pretrained(
    BASE_MODEL_NAME, local_files_only=True)
model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    local_files_only=True,
    load_in_8bit=True,
    torch_dtype=torch.float16
)
logging.info("Model and tokenizer loaded")
print_gpu_utilization()


def generate_text(system, question, response):
    return + "\nAnswer: " + response


def tokenize(text, tokenizer, max_seq_len=512, add_eos_token=True):
    result = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < max_seq_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    if add_eos_token and len(result["input_ids"]) >= max_seq_len:
        result["input_ids"][max_seq_len - 1] = tokenizer.eos_token_id
        result["attention_mask"][max_seq_len - 1] = 1

    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(sample):
    input_text = "System: " + \
        sample["system_prompt"] + "\nQuestion: " + sample["question"] + "\n"
    target_text = "Answer: " + sample["response"] + tokenizer.eos_token
    full_text = input_text + target_text
    tokenized_full_text = tokenize(full_text, tokenizer, max_seq_len=512)
    tokenized_input_text = tokenize(input_text, tokenizer, max_seq_len=512)
    input_len = len(tokenized_input_text["input_ids"]) - 1  # -1 for eos token
    tokenized_full_text["labels"] = [-100] * input_len + \
        tokenized_full_text["labels"][input_len:]
    return tokenized_full_text


# Get and tokenize dataset
train_dataset = load_dataset("Open-Orca/OpenOrca", split="train[0:80000]")
train_dataset = train_dataset.map(generate_and_tokenize_prompt)
test_dataset = load_dataset("Open-Orca/OpenOrca", split="train[80000:100000]")
test_dataset = test_dataset.map(generate_and_tokenize_prompt)
data_collator = DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
logging.info("Dataset loaded and tokenized")
print_gpu_utilization()

# Set up LoRA
peft_config = LoraConfig(
    lora_alpha=8,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
logging.info("LoRA adapter added")
print_gpu_utilization()

model.print_trainable_parameters()

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

# Train model
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=1,
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=0.00001,
        output_dir="models",
        run_name="orca-lora",
        report_to="wandb",
        num_train_epochs=1,
    ),
    data_collator=data_collator
)
trainer.train()

# Merge LoRA adapter
merged_model = model.merge_and_unload()
logging.info("LoRA adapter merged")

# Save model
model.save_pretrained(
    "models/orca-lora",
    push_to_hub=True,
    repo_id="Lienid/nous-six",
    use_auth_token=""
)
tokenizer.save_pretrained(
    "models/orca-lora",
    push_to_hub=True,
    repo_id="Lienid/nous-six",
    use_auth_token=""
)
