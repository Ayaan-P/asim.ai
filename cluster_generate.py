import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel 
import sys
import argparse 
import os 


BASE_MODEL_PATH = 'resized_gemma'
CHECKPOINT_PATH = 'results_asim_lore'

print(f"Loading tokenizer from {BASE_MODEL_PATH}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
except Exception as e:
    print(f"Error loading tokenizer from {BASE_MODEL_PATH}: {e}")
    sys.exit(1)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading base model from {BASE_MODEL_PATH}...")
try:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32 
    ).to(device) 
except Exception as e:
    print(f"Error loading base model from {BASE_MODEL_PATH}: {e}")
    sys.exit(1)


print(f"Loading PEFT adapter from checkpoint {CHECKPOINT_PATH}...")
try:
    model = PeftModel.from_pretrained(model, CHECKPOINT_PATH).to(device) 
except Exception as e:
    print(f"Error loading PEFT adapter from {CHECKPOINT_PATH}: {e}")
    sys.exit(1)


model.eval()


class StopAfterResponseToken(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stop_sequence = "\nUser:"

        self.stop_sequence_ids = tokenizer.encode(self.stop_sequence, add_special_tokens=False)

        self.stop_sequence_tensor = torch.tensor(self.stop_sequence_ids)


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if the generated sequence ends with the stop sequence token IDs
        # Ensure the generated sequence is at least as long as the stop sequence
        if input_ids[0].shape[-1] < len(self.stop_sequence_ids):
            return False

        stop_sequence_on_device = self.stop_sequence_tensor.to(input_ids.device)

        return torch.equal(input_ids[0, -len(self.stop_sequence_ids):], stop_sequence_on_device)

def generate_text(prompt, max_new_tokens=100, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.95):
    """
    Generates text based on a prompt using the loaded model.

    Args:
        prompt (str): The input text prompt.
        max_new_tokens (int): The maximum number of new tokens to generate.
        num_return_sequences (int): The number of sequences to generate.
        temperature (float): Controls the randomness of the generation. Higher values mean more random.
        top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (float): The cumulative probability of parameter for nucleus sampling.

    Returns:
        list: A list of generated text strings.
    """
    # encode the prompt
    # Format the prompt to elicit a response from "Me:"
    formatted_prompt = f"User: {prompt}\nMe:"
    input_ids = tokenizer.encode(formatted_prompt, return_tensors='pt').to(model.device) 

    # generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens, 
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True, 
            pad_token_id=tokenizer.eos_token_id, 
            stopping_criteria=StoppingCriteriaList([StopAfterResponseToken(tokenizer)]) 
        )

    #  extract the response
    generated_texts = []
    for seq in output:
        full_text = tokenizer.decode(seq, skip_special_tokens=True)

        response_start_marker = "\nMe:"
        response_start_index = full_text.find(response_start_marker)

        if response_start_index != -1:
    
            search_start_index = response_start_index + len(response_start_marker)
            stop_index = full_text.find("\nUser:", search_start_index)

            if stop_index != -1:
           
                response_text = full_text[search_start_index:stop_index].strip()
            else:

                response_text = full_text[search_start_index:].strip()
        else:

            response_text = "" 

        generated_texts.append(response_text)

    return generated_texts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a fine-tuned model on a cluster.")

    parser.add_argument("--max_new_tokens", type=int, default=100, help="The maximum number of new tokens to generate.")
    parser.add_argument("--num_return_sequences", type=int, default=3, help="The number of sequences to generate per prompt.") 
    parser.add_argument("--temperature", type=float, default=0.8, help="Controls the randomness of the generation. Higher values mean more random.") 
    parser.add_argument("--top_k", type=int, default=50, help="The number of highest probability vocabulary tokens to keep for top-k-filtering.")
    parser.add_argument("--top_p", type=float, default=0.95, help="The cumulative probability of parameter for nucleus sampling.")
    parser.add_argument("--output_file", type=str, help="Optional: Path to a file to write the generated output.")

    args = parser.parse_args()

    prompts = [
        "What are you doing this weekend?",
       
    ]

    all_generated_output = ""

    for i, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {i+1}/{len(prompts)}: '{prompt}'")
        print(f"Parameters: max_new_tokens={args.max_new_tokens}, num_return_sequences={args.num_return_sequences}, temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")

        generated_output = generate_text(
            prompt,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.num_return_sequences,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )

        all_generated_output += f"\n--- Results for Prompt {i+1}: '{prompt}' ---\n"
        for j, text in enumerate(generated_output):
            all_generated_output += f"--- Generated Sequence {j+1} ---\n"
            all_generated_output += text + "\n"
            all_generated_output += "-" * 30 + "\n"
        all_generated_output += "\n" 

    if args.output_file:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(all_generated_output)
            print(f"\nAll generated text written to {args.output_file}")
        except Exception as e:
            print(f"Error writing to output file {args.output_file}: {e}")
            
            print(all_generated_output)
    else:
        print(all_generated_output)

    print("\nGeneration complete for all prompts.")

