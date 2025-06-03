# Asimbot: WhatsApp Text Generation Model

This project aims to create a text generation model fine-tuned on WhatsApp chat data to mimic a specific individual's (Asim's) texting style.

## Project Journey: Fine-Tuning a Model on a Friend’s Chat Style

This section outlines the development process, challenges, and key learnings from building the Asimbot model.

1.  **Initial Idea & Goals**:
    *   **Goal:** Fine-tune a language model to imitate a friend’s chat style (Asim), creating a model that can generate realistic responses in casual conversation.
    *   **Stretch goal:** Implement reinforcement learning (RLHF) using friend feedback to further align the model to the desired style.

2.  **Early Experiments (GPT-2 Baseline)**:
    *   **Approach:** Initial attempts involved fine-tuning GPT-2 on raw message-response pairs.
    *   **Problem:** Coherence was poor; GPT-2 struggled to capture style or conversational flow effectively.
    *   **Conclusion:** A stronger base model and better context handling were needed.

3.  **Switching to Gemma 7B & Preprocessing Improvements**:
    *   **Model Change:** Moved to Gemma 7B with LoRA for parameter-efficient fine-tuning, leveraging its better base capabilities for conversational tasks.
    *   **Preprocessing Evolution:**
        *   Initial approach used a rigid sliding window for context.
        *   Next iteration used a large LLM API for dynamic, relevant context identification, but this was computationally expensive.
        *   **Final Strategy:** Implemented semantic retrieval by embedding messages and computing similarity to select the top N relevant context messages for each target response. This proved more efficient and produced significantly better input data for fine-tuning.

4.  **Fine-tuning Results**:
    *   Trained Gemma 7B (LoRA) on the improved dataset.
    *   **Outcomes:** Coherence significantly improved, and the model captured the tone, humor, and style of Asim’s chats better. Some variance remained, indicating potential for further improvement with RLHF.

5.  **RLHF Attempts**:
    *   **Original Plan:** Deploy an interactive bot and collect human rankings for RLHF. This was deemed too ambitious and time-consuming for the project scope.
    *   **Alternative Approach:** Explored using an LLM API to generate style-consistency scores for responses and train an RL model to maximize this score.
    *   **Implementation Issues:** Initial RL training attempts caused instability (entropy spikes, weight collapse). Despite trying adjustments to hyperparameters, the RL phase could not be fully stabilized and tuned before the project deadline.

6.  **Current Status**:
    *   The submitted model is the Gemma 7B LoRA fine-tuned on the context-selected chat dataset.
    *   RLHF attempts are considered an exploratory effort and a direction for future work, rather than a completed component with demonstrable results.

7.  **Key Learnings**:
    *   Preprocessing is critical; significant gains came from improving context selection.
    *   RLHF implementation is non-trivial and harder to stabilize than expected, especially without large-scale human feedback.
    *   LLM-based feedback scoring is a promising direction but requires better reward shaping and training stability.
    *   Future work includes developing a more robust RLHF pipeline, better reward models, and larger evaluation datasets.

## Project Components

The project is structured into several key components, reflecting the journey described above:

1.  **Data Processing (`build_whatsapp_dataset.py`, `process_chat.py`, `process_chat_llm.py`)**:
    *   **Purpose:** To extract and clean messages from raw WhatsApp chat export files (`.txt`) and structure them into conversational turns with relevant context.
    *   **Process:**
        *   The original `process_chat.py` script uses regex for basic parsing.
        *   `process_chat_llm.py` explored using an LLM for context-aware turn extraction (less used in the final pipeline due to cost).
        *   The primary script used is `build_whatsapp_dataset.py`, which implements both a simple sliding window and the more effective semantic retrieval method (embedding messages and selecting context based on similarity). This script outputs data in JSONL format, where each entry contains a `"text"` field formatted as `"{context} <|response|> {msg}"`.
        *   The output of `build_whatsapp_dataset.py` is then manually processed to replace the `<|response|>` separator with special tokens like `<USR_A>`, `<USR_B>`, and `<SYS>` to explicitly mark conversational turns and the target sender's response.
    *   **Output:** JSONL files (e.g., `asim.jsonl`, `dataset.jsonl`) containing conversational turns formatted using special tokens, such as `"<USR_A> [Message from User A] <USR_B> [Message from User B] <SYS> [Asim's Response]"`.
    *   **Why this format?** This format, utilizing distinct special tokens for different speakers and the system's response, is crucial for training a language model to act as a chatbot. By presenting the model with the conversational history (marked by `<USR_A>`, `<USR_B>`, etc.) followed by the `<SYS>` token and then Asim's actual message, we train the model to generate text that *follows* the context and the `<SYS>` prompt, effectively learning the mapping from incoming messages to Asim's typical replies. The `train_asim_lora.py` script specifically uses the position of the `<SYS>` token to mask the loss before the response part, focusing the model on generating the text that follows the system prompt.

2.  **Base Model Preparation (`prepare_gemma.py`)**:
    *   **Purpose:** To prepare the base Gemma model and tokenizer for fine-tuning by adding an initial special token and resizing the embeddings.
    *   **Process:** This script loads the original `google/gemma-7b` model and its tokenizer, adds the `<|response|>` special token, and performs an initial resizing of the token embeddings to accommodate this new token. The modified model and tokenizer are saved to the `./resized_gemma` directory. This resized model serves as the base for subsequent fine-tuning with additional special tokens.
    *   **Output:** A Gemma model and tokenizer with an expanded vocabulary, saved in `./resized_gemma`.

3.  **Model Training (`train_asim_lora.py`, `train_model.py`)**:
    *   **Purpose:** To fine-tune a pre-trained large language model (LLM) on the processed conversational turn data.
    *   **Process:** The primary training script is `train_asim_lora.py`. It loads the base LLM (the resized Gemma model from `./resized_gemma`) and its tokenizer. It then adds the specific special tokens used for training (`<USR_A>`, `<USR_B>`, `<SYS>`, and `<SEP>`) to the tokenizer and performs a *second* resizing of the model's token embeddings to accommodate these additional tokens. The script then loads and tokenizes the dataset, which is expected to be formatted using these special tokens (as produced by the manual processing step). It uses Parameter-Efficient Fine-Tuning (PEFT), specifically LoRA, to train only a small number of additional parameters. The training objective is standard causal language modeling, with loss masking applied based on the position of the `<SYS>` token in the tokenized sequence, focusing training on generating the response part.
    *   **Output:** A fine-tuned model adapter (saved in directories like `results_asim_lora/`).

4.  **Reinforcement Learning from AI Feedback (RLAIF)**:
    *   **Purpose:** (Exploratory/Future Work) To further improve the model's ability to consistently generate text that matches Asim's specific style and tone by leveraging feedback from an external LLM.
    *   **Components:**
        *   **RLAIF Dataset Generation (`generate_rlaif_dataset.py`)**: Creates a dataset of prompts, multiple model-generated responses for each prompt, and preference rankings provided by a powerful external LLM.
        *   **Reward Model Training (`train_reward_model.py`)**: Trains a separate model (the Reward Model) that can predict a scalar "reward" score for any given prompt-response pair, reflecting how well it matches the desired style based on the RLAIF dataset rankings.
        *   **RL Fine-tuning (`rl_finetune_model.py`)**: Further fine-tunes the language model using reinforcement learning, guided by the trained Reward Model, to maximize the reward signal (this phase encountered stability issues).

4.  **Text Generation (`generate_text.py`, `cluster_generate.py`)**:
    *   **Purpose:** To load the fine-tuned model and generate text responses based on user prompts.
    *   **Process:** Scripts like `generate_text.py` and `cluster_generate.py` load the base LLM and then apply the trained PEFT adapter. They take a user prompt, format it into the expected input format (which should align with the training data format using special tokens), and use the fine-tuned model to generate a continuation. Custom stopping criteria are often used to ensure the generation stops after a complete response is produced.
    *   **Output:** Generated text that mimics Asim's style in response to a given prompt.

5.  **Chatbot GUI (`chatbot_app.py`)**:
    *   **Purpose:** (Planned) To provide a user-friendly interface for interacting with the fine-tuned model.
    *   **Process:** This script would likely use a web framework (like Gradio or Streamlit) to create a simple chat interface where users can type messages and see the model's generated responses.
    *   **Status:** This is an extra criterion outlined in the implementation plan and is a future goal.

## Getting Started

(Instructions on setting up the environment, installing dependencies, and running the scripts would go here, referencing the individual script requirements).
