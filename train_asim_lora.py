#!/usr/bin/env python3
"""
LoRA‑finetune Gemma‑7B on WhatsApp‑style data.

• drops any row that has no <SYS> tag (so the model only gets user→system turns)
• hides everything up through the last <SYS> token when computing the loss
• guarantees every token‑id is within the padded vocab
"""

import argparse, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
)
from peft import LoraConfig, get_peft_model

p = argparse.ArgumentParser()
p.add_argument("--model_name", default="./resized_gemma")   # folder or HF hub id
p.add_argument("--data_path",  default="dataset.jsonl")     # JSON‑lines with "text"
p.add_argument("--out_dir",    default="./results_asim_lora")
p.add_argument("--epochs",     type=int,   default=3)
p.add_argument("--batch",      type=int,   default=1)
p.add_argument("--grad_acc",   type=int,   default=4)
p.add_argument("--block",      type=int,   default=512)     # max seq length
p.add_argument("--lr",         type=float, default=2e-4)
cfg = p.parse_args()

SPECIAL = ["<USR_A>", "<USR_B>", "<USR_C>", "<SYS>", "<SEP>"]

tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL})
if tokenizer.pad_token is None:                           # Gemma has no PAD
    tokenizer.pad_token = tokenizer.eos_token

SYS_ID = tokenizer.convert_tokens_to_ids("<SYS>")         # used in encode()

model = AutoModelForCausalLM.from_pretrained(
    cfg.model_name, torch_dtype=torch.bfloat16
)
model.resize_token_embeddings(len(tokenizer))             # NEW vocab length
model.config.use_cache = False                            # essential for LoRA

ds = load_dataset("json", data_files={"train": cfg.data_path})["train"]

def encode(batch, *, tok=tokenizer, max_len=cfg.block, sys_id=SYS_ID):
    enc = tok(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_attention_mask=True,
    )

    labels = [ids.copy() for ids in enc["input_ids"]]
    keep = []
    for i, seq in enumerate(enc["input_ids"]):
        try:                                    # keep only if <SYS> is present
            last_sys = max(j for j, t in enumerate(seq) if t == sys_id)
            labels[i][: last_sys + 1] = [-100] * (last_sys + 1)
            keep.append(i)
        except ValueError:
            continue                            # drop row with no <SYS>

    # strip rows we decided to drop
    enc = {k: [v[j] for j in keep] for k, v in enc.items()}
    enc["labels"] = [labels[j] for j in keep]
    return enc

ds = ds.map(encode, batched=True, remove_columns=["text"])
train_ds, eval_ds = ds.train_test_split(test_size=0.2, seed=42).values()

lora_cfg = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)

args = TrainingArguments(
    output_dir=cfg.out_dir,
    overwrite_output_dir=True,
    num_train_epochs=cfg.epochs,
    per_device_train_batch_size=cfg.batch,
    gradient_accumulation_steps=cfg.grad_acc,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    learning_rate=cfg.lr,
    bf16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
)

print("▶︎  training…")
trainer.train()
print("✓ done  →", cfg.out_dir)
