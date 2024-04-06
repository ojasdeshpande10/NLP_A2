from transformers import RobertaTokenizerFast, RobertaModel
import os, sys, random, re, collections, string
import numpy as np
import torch
import math
import csv
import sklearn.model_selection
from transformers import Trainer, TrainingArguments
import sklearn.metrics
import heapq
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
import matplotlib
import torch
from transformers import AdamW
from torch import nn
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from tqdm.notebook import tqdm

tokenizer = RobertaTokenizerFast.from_pretrained('distilroberta-base')

class RobertaLMBinaryClassifier(nn.Module):
    def __init__(self):
        super(RobertaLMBinaryClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('distilroberta-base')
        self.classification_head = nn.Linear(self.roberta.config.hidden_size, 1)
    def forward(self, input_ids,attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classification_head(pooled_output)
        return logits

def tokenize(batch): 
    tk_batch = tokenizer(batch['passage'],batch['question'],padding="longest",max_length=512, truncation = True)
    tk_batch['labels'] =  [int(label) for label in batch['answer']]
    return tk_batch
def train_boolq(boolq_dataset,device):
    tokenized_datasets = boolq_dataset['train'].map(tokenize, batched=True,batch_size=8)
    tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    train_loader = DataLoader(tokenized_datasets, batch_size=8)
    loss_fn = BCEWithLogitsLoss()
    model = RobertaLMBinaryClassifier().to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    for epoch in range(3):
      print(f"Epoch {epoch+1}/{3}")
      for i, batch in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].unsqueeze(1).to(device).float()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        if i%100==0:
          print(f"Batch {i}/{len(train_loader)} - Training Loss: {loss.item()/len(batch):.3f}")
    torch.save(model.state_dict(), "roberta_binary_classifier_state_dict.pt")
    return model


def validate_boolq(model,boolq_dataset,device):
    predictions = []
    actual_labels = []
    list_no=[]
    tokenized_datasets = boolq_dataset['validation'].map(tokenize, batched=True,batch_size=8)
    tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    validation_loader = DataLoader(tokenized_datasets, batch_size=8)
    with torch.no_grad(): 
        for batch in validation_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs).squeeze()
            
            if probs<0.5:
               list_no.append(1)   
            batch_predictions = (probs > 0.5).int()
            predictions.extend(batch_predictions.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
    print(len(list_no))
    accuracy = accuracy_score(actual_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(actual_labels, predictions, average=None, labels=[1, 0])
    macro_f1 = f1_score(actual_labels, predictions, average='macro')
    print(f"Accuracy: {accuracy}")
    print(f"Macro F1: {macro_f1}")
    print(f"Class-specific Precision: 'Yes': {precision[0]}, 'No': {precision[1]}")
    print(f"Class-specific Recall: 'Yes': {recall[0]}, 'No': {recall[1]}")
    print(f"Class-specific F1: 'Yes': {f1[0]}, 'No': {f1[1]}")

def main():
    boolq_dataset = load_dataset('google/boolq')
    if torch.cuda.is_available():
      device = torch.device("cuda")
      print("GPU is available")
    else:
      print("GPU is not available")
      device = torch.device("cpu")
    #emo_dataset = load_dataset('Blablablab/SOCKET', 'emobank#valence')
    model  = train_boolq(boolq_dataset, device)
    validate_boolq(model,boolq_dataset,device)
    
   
if __name__ == "__main__":
    main()