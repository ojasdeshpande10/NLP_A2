This repository contains solutions for Assignment 2 of the CSE 538.02 Natural Language Processing course at Stony Brook University, Spring 2024. The assignment is divided into three parts, focusing on various NLP tasks involving language models, including N-Gram Language Models, autoregressive models, and modifications to transformer-based models.

Repository Structure

The repository contains the following files:

a2_p1_deshpande_115837247.py

Description: Implements a Smoothed Trigram Language Model for the given datasets. The script handles the tokenization of the text data and builds a trigram language model to compute the probabilities of token sequences.

Key Features:

Tokenizes datasets using the distilgpt2 tokenizer.

Implements add-one smoothing to improve trigram probability estimates.

Outputs tokenized text and probabilities for given examples.

Usage: Run using the command:

python a2_p1_deshpande_115837247.py

Ensure that the dataset is in the correct directory before executing.

a2_p2_deshpande_115837247.py

Description: Uses an autoregressive transformer model (distil-gpt2) to evaluate zero-shot accuracy on the BoolQ dataset. It also fine-tunes the transformer model for a Boolean question-answering task.

Key Features:

Loads the distil-gpt2 model for zero-shot question answering.

Fine-tunes the model on BoolQ dataset using the Adam optimizer.

Evaluates the model's performance by computing accuracy, precision, recall, and F1 score.

Usage: Run using the command:

python a2_p2_deshpande_115837247.py

Ensure that the BoolQ dataset is accessible and that the required libraries are installed.

a2_p3_deshpande_115937247.py

Description: Implements task fine-tuning of an auto-encoding language model (distilRoberta) and modifies its architecture to experiment with attention mechanisms. The script includes training for tasks such as emotional valence prediction using EmoBank.

Key Features:

Fine-tunes distilRoberta for Boolean classification and emotional valence regression tasks.

Modifies attention layers, including removing residual links and sharing key/query weights.

Compares model performance across different architectures.

Usage: Run using the command:

python a2_p3_deshpande_115937247.py

Ensure all required datasets (BoolQ and EmoBank) are accessible and set up properly.

train.jsonl

Description: Training dataset in JSON Lines format used for model training and evaluation. Each line contains a separate JSON object for efficient data loading and processing.

Usage: Ensure that this file is available in the same directory as the Python scripts. The file will be automatically loaded by the scripts as needed.

Dependencies

The scripts require the following libraries:

Python 3.8+

PyTorch 2.1.1+

Numpy 1.22+

Transformers (Hugging Face)

Sklearn

Install dependencies using the command:

pip install -r requirements.txt

Execution Order

It is recommended to execute the scripts in the following order:

a2_p1_deshpande_115837247.py: To build and evaluate the N-Gram Language Model.

a2_p2_deshpande_115837247.py: To work with the autoregressive transformer model.

a2_p3_deshpande_115937247.py: To fine-tune transformer models and experiment with modifications.

Datasets Used

BoolQ: A Boolean question-answering dataset with naturally occurring questions, used for both zero-shot evaluation and fine-tuning.

EmoBank-Valence: A dataset with emotional valence scores used for regression tasks.

Datasets are loaded using the Hugging Face datasets library:

from datasets import load_dataset
boolq_dataset = load_dataset('google/boolq')
emo_dataset = load_dataset('Blablablab/SOCKET', 'emobank#valence')

Tips for Running

Ensure a compatible environment with Python 3.8 or later.

Utilize GPU (e.g., NVidia T4) to expedite fine-tuning tasks.

Each script is designed to run independently, but the output from previous parts may be useful for comparison in later parts.
