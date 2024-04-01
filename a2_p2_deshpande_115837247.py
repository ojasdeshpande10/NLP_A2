import os, sys, random, re, collections, string

import numpy as np

import torch

import math
import time

import csv

import sklearn.model_selection
import sklearn.metrics
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, AutoModel
from datasets import load_dataset
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence


class LanguageModelingClass:

  def __init__(self, model_name):
    self.model_name = model_name
    self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
    self.model = GPT2LMHeadModel.from_pretrained(model_name)
    if torch.cuda.is_available():
      self.device = torch.device("cuda")
      print("GPU is available")
    else:
      self.device = torch.device("cpu")
      print("GPU not available, using CPU instead")
    self.model = self.model.to(self.device)
  def fineTuneLM(self, dataset):
    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    self.model.resize_token_embeddings(len(self.tokenizer))
    def tokenize_and_combine_examples(examples):
      passage, question, answer = examples['passage'], examples['question'], examples['answer']
      max_passage_length = 1024-len(question)-3
      if answer:
        text_ans ='yes'
      else:
        text_ans='no'
      truncated_text_entry = passage[:max_passage_length] + '\n' + question + '?\n' + text_ans
      inputs = self.tokenizer(truncated_text_entry, return_tensors='pt').to(self.device)
      # Tokenize combined text
      return inputs
    optimizer = AdamW(self.model.parameters(), lr=5e-5, weight_decay=0.001)
    self.model.train()
    epochs = 1
    print("Starting fineTuning")
    for epoch in range(epochs):
      total_loss = 0.0  # Initialize total loss for the epoch
      num_entries = 0   
      for entry in dataset['train']:
        inputs = tokenize_and_combine_examples(entry)
        outputs = self.model(**inputs,labels=inputs["input_ids"])
        loss = outputs.loss  
        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad() 
        total_loss += loss.item()
        num_entries+=1
        if num_entries % 1000 == 0:
          print(f"Epoch {epoch}, Entry {num_entries}: Loss {loss.item()}", flush=True)
      avg_loss = total_loss / num_entries
      print(f"Epoch: {epoch}, Average Loss: {avg_loss}", flush=True)
  def getValidationScore(self, dataset):
    predictions = []
    actual_labels = []
    yes_token_id = self.tokenizer.encode('yes', add_special_tokens=False)[0]
    no_token_id = self.tokenizer.encode('no', add_special_tokens=False)[0]
    for split in dataset:
      if split == 'validation':
        for entry in dataset[split]:
          passage, question = entry['passage'], entry['question']
          max_passage_length = 1024-len(question)-3
          truncated_text_entry = passage[:max_passage_length] + '\n' + question + '?\n'
          inputs = self.tokenizer(truncated_text_entry, return_tensors='pt').to(self.device)
          with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            yes_logits = logits[:, -1, yes_token_id]
            no_logits = logits[:, -1, no_token_id]
            softmax_scores = torch.softmax(torch.stack((yes_logits, no_logits)), dim=0)
            yes_prob, no_prob = softmax_scores.cpu().numpy()
          prediction = 'yes' if yes_prob > no_prob else 'no'
          predictions.append(prediction)
          actual_labels.append('yes' if entry['answer'] else 'no')
    accuracy = accuracy_score(actual_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(actual_labels, predictions, average=None, labels=['yes', 'no'])
    macro_f1 = f1_score(actual_labels, predictions, average='macro')
    print(f"Accuracy: {accuracy}")
    print(f"Macro F1: {macro_f1}")
    print(f"Class-specific Precision: 'Yes': {precision[0]}, 'No': {precision[1]}")
    print(f"Class-specific Recall: 'Yes': {recall[0]}, 'No': {recall[1]}")
    print(f"Class-specific F1: 'Yes': {f1[0]}, 'No': {f1[1]}")


def main():


  boolq_dataset = load_dataset('google/boolq')
  lm = LanguageModelingClass('distilgpt2')
  # Zero shot accuracy without finetuning
  start_time = time.time()
  lm.getValidationScore(boolq_dataset)
  lm.fineTuneLM(boolq_dataset)
  lm.getValidationScore(boolq_dataset)
  end_time = time.time()
  print("Time_Required= ",end_time-start_time)
  #print(gpt2_tokenizer.decode(50000))
if __name__ == "__main__":
    main()

