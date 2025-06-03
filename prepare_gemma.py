from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "google/gemma-7b"
OUTPUT_DIR = "./resized_gemma"
SPECIAL_TOKEN = "<|response|>"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_special_tokens({'additional_special_tokens': [SPECIAL_TOKEN]})

print("Saving tokenizer...")
tokenizer.save_pretrained(OUTPUT_DIR)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)

print("Resizing token embeddings...")
model.resize_token_embeddings(len(tokenizer))

print("Saving model...")
model.save_pretrained(OUTPUT_DIR)

print("Done.")
