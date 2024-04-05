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
from tqdm import tqdm

class RobertaLMMod(nn.Module):
    def __init__(self):
        super(RobertaLMMod, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.classification_head = nn.Linear(self.roberta.config.hidden_size, 1)
    def forward(self, input_ids,attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classification_head(pooled_output)
        return logits
def main():
    boolq_dataset = load_dataset('google/boolq')
    if torch.cuda.is_available():
      device = torch.device("cuda")
      print("GPU is available")
    else:
      print("GPU is not available")
      device = torch.device("cpu")
    #emo_dataset = load_dataset('Blablablab/SOCKET', 'emobank#valence')
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    def tokenize(batch): 
      tk_batch = tokenizer(batch['passage'],batch['question'],padding="longest",max_length=512, truncation = True)
      tk_batch['labels'] =  [int(label) for label in batch['answer']]
      return tk_batch
    tokenized_datasets = boolq_dataset['train'].map(tokenize, batched=True,batch_size=4)
    tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    train_loader = DataLoader(tokenized_datasets, batch_size=4)
    loss_fn = BCEWithLogitsLoss()
    model = RobertaLMMod().to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    for epoch in range(3):
      progress_bar = tqdm(train_loader, desc="Epoch {:1d}".format(epoch+1), leave=False, disable=False)
      for batch in train_loader:
          optimizer.zero_grad()
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          labels = batch['labels'].unsqueeze(1).to(device).float()
          outputs = model(input_ids, attention_mask)
          loss = loss_fn(outputs, labels)
          loss.backward()
          optimizer.step()
          progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))}) 
            
if __name__ == "__main__":
    main()