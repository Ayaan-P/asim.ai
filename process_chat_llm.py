import argparse
import json
import random
import re
import os
import google.generativeai as genai
import time
import sys

# Define a special token to separate context from response
RESPONSE_TOKEN = "<|response|>"

# --- Configuration ---
# Model to use for parsing
PARSER_LLM_NAME = "gemini-2.0-flash-thinking-exp-01-21" # Using a capable model for parsing

# --- Helper Functions ---

def clean_message(message):
    """Removes unwanted system messages and artifacts from a message."""

    unwanted_patterns = [
        r'‎Messages and calls are end-to-end encrypted\.',
        r'‎You created group “.*”',
        r'‎.* joined using your invite',
        r'‎.* joined using this group\'s invite link',
        r'‎document omitted',
        r'‎Voice call',
        r'‎Call failed',
        r'‎.* missed a voice call',
        r'‎.* missed a video call',
        r'‎.* left',
        r'‎.* changed the group description',
        r'‎.* changed this group\'s settings to allow only admins to edit this group\'s settings',
        r'‎.* changed this group\'s settings to allow only admins to send messages to this group',
        r'‎.* changed the group icon',
        r'‎.* changed the group name',
        r'‎.* added',
        r'‎.* removed',
        r'‎.* became an admin',
        r'‎.* is no longer an admin',
        r'‎.* changed their phone number to a new number\. Tap to add them as a contact\.',
        # Add more patterns as observed in the chat logs
    ]
    cleaned_message = message
    for pattern in unwanted_patterns:
        cleaned_message = re.sub(pattern, '', cleaned_message)

    # Remove leading/trailing whitespace
    cleaned_message = cleaned_message.strip()

    return cleaned_message


def parse_chat_with_llm(chat_chunk, sender_name, api_key):
    """
    Uses an LLM to parse a chunk of chat and extract conversational turns.
    """
    if not api_key:
        print("Error: API key not configured for parser LLM.")
        return []

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(PARSER_LLM_NAME)


        parser_prompt = f"""
You are a chat log parser. Your task is to read a section of a WhatsApp chat log and identify conversational turns where a specific sender responds to messages from other participants.

The goal is to create a dataset to finetune an llm to speak like the target sender. Therefore we need a dataset in a prompt response format.

A conversational turn consists of:
1.  Context: One or more messages from participants *other than* the target sender, immediately preceding the target sender's message. **Crucially, remove any timestamps and sender names from the beginning of each line in the context.** as the user will not be prompting th emodel with these details
2.  Response: A single message from the target sender that is a response to the preceding context.

Your output should be a list of these conversational turns, formatted exactly as:
[Cleaned Context from others] {RESPONSE_TOKEN} [Target Sender's Response]

Each turn should be on a new line. Do NOT include any other text, explanations, or formatting in your output, only the extracted turns.

Target Sender: {sender_name}
Response Token: {RESPONSE_TOKEN}

Here is the chat log section:
---
{chat_chunk}
---

Extract the conversational turns for the Target Sender in the specified format.
"""

        response = model.generate_content(parser_prompt)


        extracted_turns = []
        if response and response.text:
            for line in response.text.strip().split('\n'):
                line = line.strip()
                if line and RESPONSE_TOKEN in line:

                    parts = line.split(RESPONSE_TOKEN, 1)
                    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                         extracted_turns.append(line)
                    else:
                         print(f"Warning: LLM returned a line with the token but incorrect format: {line}")
                elif line:
                     print(f"Warning: LLM returned a line without the response token: {line}")


        return extracted_turns

    except Exception as e:
        print(f"Error calling parser LLM API: {e}")

        return []

# --- Main Script Logic ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process WhatsApp chat files using an LLM to extract conversational turns.')
    parser.add_argument('file_paths', metavar='FILE', type=str, nargs='+',
                        help='Paths to the raw WhatsApp chat files (e.g., asim_messages.txt).')
    parser.add_argument('--sender', type=str, required=True,
                        help='The name of the sender whose messages are the target responses.')
    parser.add_argument('--output', type=str, required=True,
                        help='The path to the output file to save the extracted turns (e.g., asim_messages-context.txt).')
    parser.add_argument('--chunk_size', type=int, default=50,
                        help='Number of raw chat lines to process in each LLM call.')


    args = parser.parse_args()

    api_key = "AIzaSyBTvrrYDH0a7LWhH_N0yQ3cqOCu47RSqJQ"


    all_raw_lines = []
    for file_path in args.file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found at {file_path}. Skipping.")
            continue
        print(f"Reading raw chat file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            all_raw_lines.extend(f.readlines())

    if not all_raw_lines:
        print("No raw chat data loaded. Exiting.")
        sys.exit(1)

    print(f"Loaded {len(all_raw_lines)} raw chat lines.")

    extracted_conversational_turns = []

    # Process chat in chunks
    for i in range(0, len(all_raw_lines), args.chunk_size):
        chunk = "".join(all_raw_lines[i : i + args.chunk_size])
        print(f"Processing chunk {i//args.chunk_size + 1} (lines {i}-{min(i+args.chunk_size, len(all_raw_lines))-1})...")

        # Use LLM to parse the chunk
        turns_in_chunk = parse_chat_with_llm(chunk, args.sender, api_key)
        extracted_conversational_turns.extend(turns_in_chunk)

        # Print an example of the extracted dataset after the first chunk
        if i == 0 and turns_in_chunk:
            print("\n--- Example Extracted Turn (after first chunk) ---")
            print(turns_in_chunk[0])
            print("--------------------------------------------------\n")


        print(f"  Extracted {len(turns_in_chunk)} turns from this chunk. Total turns: {len(extracted_conversational_turns)}")

        
        time.sleep(2) 

    # Write extracted turns to the output file
    print(f"\nWriting extracted conversational turns to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for turn in extracted_conversational_turns:
            f.write(turn + '\n')

    print(f"LLM-based chat processing complete. Saved {len(extracted_conversational_turns)} conversational turns to {args.output}")
