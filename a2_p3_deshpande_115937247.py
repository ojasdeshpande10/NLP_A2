from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import os, sys, random, re, collections, string

import numpy as np

import torch

import math

import csv

import sklearn.model_selection
from transformers import Trainer, TrainingArguments
import sklearn.metrics
import heapq

import matplotlib
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from collections import defaultdict


def main():

    boolq_dataset = load_dataset('google/boolq')
    emo_dataset = load_dataset('Blablablab/SOCKET', 'emobank#valence')
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    def tokenize(batch): 
      tk_batch = tokenizer(batch['passage'],batch['question'],padding="max_length", truncation = True)
      tk_batch['labels'] =  [int(label) for label in batch['answer']]
      return tk_batch
    tokenized_datasets = boolq_dataset.map(tokenize, batched=True)
    tokenized_datasets = tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    def compute_metrics(p):
        predictions = np.argmax(p.predictions, axis=1)
        accuracy = sklearn.metrics.accuracy_score(p.label_ids, predictions)
        return {"accuracy": accuracy}
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        learning_rate=5e-5,
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.evaluate()




if __name__ == "__main__":

    main()
