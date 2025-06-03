# Asimbot Project Implementation Plan

This document outlines the steps for implementing the Asimbot text generation project, from data processing to model deployment and integration.

## Phase 1: Data Processing

**Goal:** Extract and structure conversational turns from WhatsApp chat logs using both regex-based and LLM-based methods.

1.  **Develop Data Processing Scripts:**
    *   **Regex-based (`process_chat.py`):** Create/refine a Python script (`process_chat.py`) that accepts input file paths (WhatsApp chat `.txt` files) and the target sender name ('Asim') as command-line arguments. Implement logic to read, parse, filter, and clean messages, potentially structuring them into simple context-response pairs or just extracting target sender messages.
    *   **LLM-based (`process_chat_llm.py`):** Develop a Python script (`process_chat_llm.py`) that uses an external LLM to parse chat chunks and identify conversational turns, extracting pairs of context (messages from others) and the target sender's response. This script should accept input file paths, the target sender name, and an API key for the LLM.
    *   Implement cleaning steps in both scripts to remove unwanted elements such as system messages.

2.  **Execute Data Processing Scripts:**
    *   Run the `process_chat_llm.py` script using the `execute_command` tool to process the initial chat files (e.g., `WhatsApp Chat - Asim/_chat.txt`) and generate the LLM-processed dataset (`asim_messages-llm.txt`).
    *   (Optional) Run `process_chat.py` if a regex-based dataset is also desired for comparison or other purposes.
    *   Verify the output file(s) (e.g., `asim_messages-llm.txt`) to ensure conversational turns are correctly extracted and formatted.

## Phase 2: Model Development and Training

**Goal:** Select, prepare data for, and fine-tune a pre-trained transformer model on the processed Asim messages dataset using available A100 GPUs.

1.  **Model Selection:**
    *   Research and select a suitable pre-trained decoder-only transformer model for text generation from a library like Hugging Face `transformers`. Consider models like GPT-2 (various sizes) or similar architectures known for conversational text generation.
    *   Factors for selection: model size (balancing performance potential with A100 memory capacity), availability of a compatible tokenizer, and ease of fine-tuning.

2.  **Tokenization and Dataset Preparation:**
    *   Load the tokenizer corresponding to the selected pre-trained model.
    *   Tokenize the `asim_messages.txt` dataset using the loaded tokenizer.
    *   Prepare the tokenized data for training. This typically involves:
        *   Concatenating messages into longer sequences or processing them as individual examples.
        *   Creating input sequences and corresponding labels (which are often the next token in the sequence for language modeling).
        *   Handling padding and attention masks as required by the model architecture.
        *   Splitting the dataset into training and validation sets (e.g., 80/20 split).

3.  **Fine-tuning Setup:**
    *   Load the selected pre-trained model weights.
    *   Configure the training environment to utilize the A100 GPUs (e.g., using PyTorch with CUDA or TensorFlow with GPU support).
    *   Define the training parameters:
        *   Optimizer (e.g., AdamW).
        *   Loss function (e.g., Cross-Entropy Loss).
        *   Learning rate (start with a small value, typical for fine-tuning).
        *   Batch size (maximize based on A100 VRAM).
        *   Number of training epochs (start with a few, monitor validation loss).
        *   Weight decay, learning rate scheduler, and other regularization techniques.

4.  **Fine-tuning Execution:**
    *   Implement the training loop.
    *   Train the model on the prepared training dataset.
    *   Monitor training progress (loss, potentially perplexity).
    *   Evaluate the model periodically on the validation set to prevent overfitting and guide hyperparameter tuning.
    *   Save the fine-tuned model weights.

## Phase 3: Model Evaluation

**Goal:** Assess the performance and quality of the fine-tuned text generation model.

1.  **Qualitative Evaluation:**
    *   Generate text samples using the fine-tuned model with various prompts.
    *   Manually review the generated text for:
        *   Fluency and coherence.
        *   Relevance to the prompt.
        *   Similarity to Asim's conversational style, tone, and common phrases.
        *   Absence of repetitive or nonsensical output.
2.  **Quantitative Evaluation (Optional but Recommended):**
    *   Calculate metrics like perplexity on a held-out test set (if the dataset size allows for a separate test set).
    *   While perplexity is useful, qualitative evaluation is often more critical for assessing style-specific generation.

## Phase 4: Reinforcement Learning from AI Feedback (RLAIF)

**Goal:** Further improve the model's style adherence using AI feedback.

1.  **Develop RLAIF Dataset Generation Script:**
    *   Create a Python script (`generate_rlaif_dataset.py`) that loads the processed conversational turns (e.g., from `asim_messages-llm.txt`).
    *   Load the fine-tuned model from Phase 2.
    *   Select prompts from the conversational turns.
    *   Use the fine-tuned model to generate multiple responses for each selected prompt.
    *   Call an external LLM API (the evaluator LLM) to rank the generated responses based on style adherence and relevance, providing examples of the target style (full conversational turns) as context to the evaluator LLM.
    *   Save the prompts, generated responses, and the LLM-provided rankings to an output file (e.g., a JSONL file).

2.  **Develop Reward Model Training Script:**
    *   Create a script to train a Reward Model on the RLAIF dataset generated in the previous step. This model will learn to predict a score indicating how well a response matches the desired style.

3.  **Develop RL Fine-tuning Script:**
    *   Create a script to fine-tune the language model using reinforcement learning (e.g., PPO), guided by the trained Reward Model. The model will be updated to maximize the reward signal, improving its style alignment.

4.  **Execute RLAIF Steps:**
    *   Run the `generate_rlaif_dataset.py` script using `execute_command` to create the RLAIF dataset.
    *   Run the Reward Model training script.
    *   Run the RL fine-tuning script.

## Phase 5: Chatbot GUI Development (Extra Criteria)

**Goal:** Create a user interface to interact with the fine-tuned Asimbot model.

1.  **Select GUI Framework:** Choose a suitable framework (e.g., Gradio, Streamlit, Flask/HTML/JS). Gradio or Streamlit are recommended for rapid prototyping.
2.  **Develop Interface:**
    *   Create a simple web-based interface.
    *   Include a text input area for the user to type prompts.
    *   Include a button to trigger text generation.
    *   Include a display area to show the generated response from the Asimbot model.
3.  **Integrate Model:**
    *   Load the fine-tuned model within the GUI application.
    *   Implement logic to take the user's input, pass it to the model for generation, and display the model's output.

## Phase 5: WhatsApp Integration (Extra Criteria - Exploration/Proof-of-Concept)

**Goal:** Explore and potentially implement a basic integration with WhatsApp.

1.  **Research WhatsApp APIs:** Investigate the official WhatsApp Business API and its requirements (business account, verification, hosting).
2.  **Explore Alternatives (with caution):** Research third-party libraries or unofficial methods for WhatsApp interaction, understanding the associated risks (ToS violations, instability, security).
3.  **Proof-of-Concept:** If feasible and within project scope/timeline, attempt to build a minimal proof-of-concept integration (e.g., receiving a message and sending a hardcoded or simple model response). Full, robust integration is likely beyond the scope of a course project.

## Phase 6: Documentation and Presentation

**Goal:** Document the project and prepare for presentation.

1.  **Code Documentation:** Add comments to the code, explaining key functions and logic.
2.  **Project Report/Documentation:** Write a report detailing:
    *   The project goal and motivation.
    *   Data sources and processing steps.
    *   Model selection and architecture explanation.
    *   Fine-tuning process and hyperparameters.
    *   Evaluation results (qualitative and quantitative).
    *   GUI implementation details.
    *   WhatsApp integration exploration/status.
    *   Challenges encountered and lessons learned.
    *   Future work.
3.  **Presentation Materials:** Prepare slides or other materials for presenting the project.

## Tools to be Used

*   **`execute_command`:** For running the data processing script, installing libraries (pip), and potentially running the GUI server.
*   **`write_to_file`:** For creating the Python data processing script and other code files (model training script, GUI code).
*   **`read_file`:** To review code files if needed during development or debugging (though direct reading of large chat files is avoided).
*   **`replace_in_file`:** For making targeted modifications to existing code files.
*   **`browser_action`:** Potentially for testing the web-based GUI.
