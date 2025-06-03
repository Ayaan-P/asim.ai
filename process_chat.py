import argparse
import re
import os

# Define a special token to separate context from response
RESPONSE_TOKEN = "<|response|>"

def clean_message(message):
    """Removes unwanted system messages and artifacts from a message."""
    # List of patterns to remove
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

def process_chat_files(file_paths, sender_name, output_file):
    """
    Processes WhatsApp chat files to extract conversational turns
    (context from others followed by target sender's response).

    Args:
        file_paths (list): A list of paths to the WhatsApp chat files.
        sender_name (str): The name of the sender whose messages are the target responses.
        output_file (str): The path to the output file to save the extracted turns.
    """
    conversational_turns = []
    message_pattern = re.compile(r'^\[\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}:\d{2} [AP]M\] (.*?): (.*)$')
    current_context_messages = []
    current_message = None

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found at {file_path}. Skipping.")
            continue

        print(f"Processing file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue # Skip empty lines

                match = message_pattern.match(line)
                if match:
                    # Process the previous message if it exists
                    if current_message:
                        cleaned_msg = clean_message(current_message['content'])
                        if cleaned_msg:
                            if current_message['sender'] == sender_name:
                                # This is the target sender's response
                                if current_context_messages: # Only add if there's context
                                    context_text = "\n".join(current_context_messages)
                                    conversational_turns.append(f"{context_text} {RESPONSE_TOKEN} {cleaned_msg}")
                                current_context_messages = [] # Reset context after target sender's message
                            else:
                                # This is a message from another sender, add to context
                                current_context_messages.append(cleaned_msg)

                    # Start a new message
                    sender, content = match.groups()
                    current_message = {'sender': sender.strip(), 'content': content.strip()}
                elif current_message:
                    # Continuation of the previous message
                    current_message['content'] += '\n' + line
                # Lines before the first message or unparsable lines are ignored

            # Process the very last message in the file
            if current_message:
                cleaned_msg = clean_message(current_message['content'])
                if cleaned_msg and current_message['sender'] == sender_name:
                     if current_context_messages:
                        context_text = "\n".join(current_context_messages)
                        conversational_turns.append(f"{context_text} {RESPONSE_TOKEN} {cleaned_msg}")


    print(f"Extracted {len(conversational_turns)} conversational turns.")

    # Write extracted turns to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for turn in conversational_turns:
            f.write(turn + '\n')

    print(f"Conversational turns saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process WhatsApp chat files to extract conversational turns.')
    parser.add_argument('file_paths', metavar='FILE', type=str, nargs='+',
                        help='Paths to the WhatsApp chat files.')
    parser.add_argument('--sender', type=str, required=True,
                        help='The name of the sender whose messages are the target responses.')
    parser.add_argument('--output', type=str, required=True,
                        help='The path to the output file to save the extracted turns.')

    args = parser.parse_args()

    process_chat_files(args.file_paths, args.sender, args.output)
