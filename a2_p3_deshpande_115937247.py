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
import matplotlib.pyplot as plt
from torch import nn
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error


tokenizer = RobertaTokenizerFast.from_pretrained('distilroberta-base')

class RobertaLMMod(nn.Module):
    def __init__(self):
        super(RobertaLMMod, self).__init__()
        self.roberta = RobertaModel.from_pretrained('distilroberta-base')
        self.classification_head = nn.Linear(self.roberta.config.hidden_size, 1)
    def forward(self, input_ids,attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classification_head(pooled_output)
        return logits


class RobertaPoolerMod(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Instead of pooling the first token we are going to take the mean of all the hidden_states
        mean_hidden_states = torch.mean(hidden_states, dim=1)
        pooled_output = self.dense(mean_hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class RobertaLMModImproved(nn.Module):
    def __init__(self):
        super(RobertaLMMod, self).__init__()
        self.roberta = RobertaModel.from_pretrained('distilroberta-base')
        self.config = self.roberta.config
        self.changePooler(self.config)
        self.classification_head = nn.Linear(self.roberta.config.hidden_size, 1)
    def forward(self, input_ids,attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classification_head(pooled_output)
        return logits
    def changePooler(self, config):
        self.roberta.pooler = RobertaPoolerMod(config)


class DistilRBRand(nn.Module):
    def __init__(self):
        super(DistilRBRand, self).__init__()
        self.roberta = RobertaModel.from_pretrained('distilroberta-base')
        self._randomize_weights(self.roberta)
        self.classification_head = nn.Linear(self.roberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classification_head(pooled_output)
        return logits

    def _randomize_weights(self, model):
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=1.0)
                # initializing biases to zeros
                if module.bias is not None:
                    module.weight.data.normal_(mean=0.0, std=1.0)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.fill_(10.0)
                module.weight.data.fill_(1.0)
class DistilRBKQ(nn.Module):
    def __init__(self):
        super(DistilRBKQ, self).__init__()
        self.roberta = RobertaModel.from_pretrained('distilroberta-base')
        self._modify_attention(self.roberta)
        self.classification_head = nn.Linear(self.roberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classification_head(pooled_output)
        return logits
    def _modify_attention(self, model):
        for i in [-2, -1]:
            layer = model.encoder.layer[i]
            query = layer.attention.self.query
            key = layer.attention.self.key
            # Averaging the weights
            mean_weight = torch.mean(torch.stack([query.weight.data, key.weight.data]), dim=0)
            query.weight.data = mean_weight
            key.weight.data = mean_weight
class RobertaNoResidualOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states) # removing the residual link
        return hidden_states
class RobertaNoResidualSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states) # No residual link
        return hidden_states

class DistilRBnores(nn.Module):
    def __init__(self):
        super(DistilRBnores, self).__init__()
        self.roberta = RobertaModel.from_pretrained('distilroberta-base')
        self.config = self.roberta.config
        self._remove_residual(self.roberta)
        self.classification_head = nn.Linear(self.roberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classification_head(pooled_output)
        return logits
    def _remove_residual(self, model):
        for i in [-2, -1]:
            layer = model.encoder.layer[i]
            layer.attention.output = RobertaNoResidualSelfOutput(config=self.config)
            layer.output = RobertaNoResidualOutput(config=self.config)
def tokenize(batch): 
    tk_batch = tokenizer(batch['passage'],batch['question'],padding="longest",max_length=512, truncation = True)
    tk_batch['labels'] =  [int(label) for label in batch['answer']]
    return tk_batch
def tokenize1(batch): 
    tk_batch = tokenizer(batch['text'],padding="longest",max_length=512, truncation = True)
    tk_batch['labels'] =  batch['label']
    return tk_batch

def train_boolq(model, boolq_dataset,device):
    tokenized_datasets = boolq_dataset['train'].map(tokenize, batched=True,batch_size=16)
    tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    train_loader = DataLoader(tokenized_datasets, batch_size=16)
    loss_fn = BCEWithLogitsLoss()
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    train_losses=[]
    for epoch in range(3):
      epoch_losses=[]
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
        epoch_losses.append(loss.item())
      epoch_loss = sum(epoch_losses) / len(epoch_losses)   
      print(f"Epoch {epoch+1}, Loss: {epoch_loss}")
      train_losses.append(epoch_loss)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()
def validate_boolq(model,boolq_dataset,device,only_acc):
    predictions = []
    actual_labels = []
    list_no=[]
    tokenized_datasets = boolq_dataset['validation'].map(tokenize, batched=True,batch_size=16)
    tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    validation_loader = DataLoader(tokenized_datasets, batch_size=16)
    with torch.no_grad(): 
        for batch in validation_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs).squeeze()
            batch_predictions = (probs > 0.5).int()
            predictions.extend(batch_predictions.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
    print(len(list_no))
    accuracy = accuracy_score(actual_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(actual_labels, predictions, average=None, labels=[1, 0])
    macro_f1 = f1_score(actual_labels, predictions, average='macro')
    if only_acc:
        print(f"Accuracy: {accuracy}")
        print(f"Macro F1: {macro_f1}")
        print(f"Class-specific Precision: 'Yes': {precision[0]}, 'No': {precision[1]}")
        print(f"Class-specific Recall: 'Yes': {recall[0]}, 'No': {recall[1]}")
        print(f"Class-specific F1: 'Yes': {f1[0]}, 'No': {f1[1]}")
    return accuracy, macro_f1
def train_emobank(model, dataset, device):
    tokenized_datasets = dataset['train'].map(tokenize1, batched=True,batch_size=8)
    tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    train_loader = DataLoader(tokenized_datasets, batch_size=8)
    loss_fn = torch.nn.MSELoss()
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    train_losses=[]
    for epoch in range(3):
      epoch_losses=[]
      for i, batch in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].unsqueeze(1).to(device).float()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
      epoch_loss = sum(epoch_losses) / len(epoch_losses)   
      print(f"Epoch {epoch+1}, Loss: {epoch_loss}")
      train_losses.append(epoch_loss)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()
def validate_emobank(model,dataset, device):
    model.eval()
    predictions = []
    true_labels = []
    tokenized_datasets = dataset.map(tokenize1, batched=True,batch_size=8)
    tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    validate_loader = DataLoader(tokenized_datasets, batch_size=8)
    with torch.no_grad():
        for batch in validate_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device).float().unsqueeze(1)
            outputs = model(input_ids, attention_mask).squeeze()
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    mae = mean_absolute_error(true_labels, predictions)
    pearson_corr, _ = pearsonr(true_labels.flatten(), predictions.flatten())
    return mae, pearson_corr
def main():
    boolq_dataset = load_dataset('google/boolq')
    emo_dataset = load_dataset('Blablablab/SOCKET', 'emobank#valence')
    if torch.cuda.is_available():
      device = torch.device("cuda")
      print("GPU is available")
    else:
      print("GPU is not available")
      device = torch.device("cpu")

    print("******CheckPoint 3.1*******")
    model_3_1 = RobertaLMMod().to(device)
    train_boolq(model_3_1,boolq_dataset, device)
    accuracy_3_1, f1_3_1 = validate_boolq(model_3_1,boolq_dataset,device,1)
    model_3_1 = model_3_1.to('cpu')
    print("******CheckPoint 3.2*****")
    model_3_2 = RobertaLMMod().to(device)
    train_emobank(model_3_2, emo_dataset, device)
    mae, pearson_r = validate_emobank(model_3_2,emo_dataset['validation'],device)
    print(f"Validation   mae: {mae}  pearson: {pearson_r}")
    mae_3_2, pearson_r_3_2 = validate_emobank(model_3_2,emo_dataset['test'],device)
    print(f"Test   mae: {mae_3_2}  pearson: {pearson_r_3_2}")
    model_3_2 = model_3_2.to('cpu')
    models_boolq = [
        DistilRBRand().to(device),
        DistilRBKQ().to(device),
        DistilRBnores().to(device)
    ]
    models_emo = [
        DistilRBRand().to(device),
        DistilRBKQ().to(device),
        DistilRBnores().to(device)
    ]

    
    # finetuning key-query model on both datasets
    train_boolq(models_boolq[1],boolq_dataset, device)
    train_emobank(models_emo[1], emo_dataset, device)
    # finetuning key-query model on both datasets on
    train_boolq(models_boolq[2],boolq_dataset, device)
    train_emobank(models_emo[2], emo_dataset, device)
    # 3.3.1

    accuracy_3_3_1, f1_3_3_1 = validate_boolq(models_boolq[0], boolq_dataset,device,0)
    mae_3_3_1, pearson_r_3_3_1 = validate_emobank(models_emo[0],emo_dataset['test'],device)



    # 3.3.2
    accuracy_3_3_2, f1_3_3_2 = validate_boolq(models_boolq[1],boolq_dataset,device,0)
    mae_3_3_2, pearson_r_3_3_2 = validate_emobank(models_emo[1],emo_dataset['test'],device)
    accuracy_3_3_3, f1_3_3_3 = validate_boolq(models_boolq[2],boolq_dataset,device,0)
    mae_3_3_3, pearson_r_3_3_3 = validate_emobank(models_emo[2],emo_dataset['test'],device)
    print("*****CheckPoint 3.4********")
    print("boolqvalidation-dataset:")
    print(f"distilroberta:   overall acc: {accuracy_3_1} f1: {f1_3_1}")
    print(f"distilRB-rand:   overall acc: {accuracy_3_3_1} f1: {f1_3_3_1}")
    print(f"distilRB-KQ:   overall acc: {accuracy_3_3_2} f1: {f1_3_3_2}")
    print(f"distilRB-nores:   overall acc: {accuracy_3_3_3} f1: {f1_3_3_3}")

    print("emobank-dataset")
    print(f"distilroberta:   mae: {mae_3_2} r: {pearson_r_3_2}")
    print(f"distilRB-rand:   mae: {mae_3_3_1} r: {pearson_r_3_3_1}")
    print(f"distilRB-KQ:   mae: {mae_3_3_2} r: {pearson_r_3_3_2}")
    print(f"distilRB-nores:   mae: {mae_3_3_3} r: {pearson_r_3_3_3}")


    print("******CheckPoint 3.5*******")
    model_3_5 = RobertaLMModImproved().to(device)
    train_boolq(model_3_5,boolq_dataset, device)
    accuracy_3_5, f1_3_5 = validate_boolq(model_3_5,boolq_dataset,device,1)
if __name__ == "__main__":
    main()