import os, sys, random, re, collections, string

import numpy as np

import torch

import math

import csv

import sklearn.model_selection
import sklearn.metrics
import heapq

import matplotlib
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset



def tokenize_dataset_field(dataset, tokenizer):
    tokenized_dataset={}
    for split in dataset:
        for entry in dataset[split]:
            question = tokenizer.encode(entry['question'])
            passage =  tokenizer.encode(entry[
            if entry['answer'] == 'True':
                answer = 'Yes'
            else:
                answer = 'No'

            tokenize_question = tokenizer

            
            



def main():
    boolq_dataset = load_dataset('google/boolq')
    emo_dataset = load_dataset('Blablablab/SOCKET', 'emobank#valence')
    gpt2_tokenizer = PreTrainedTokenizerFast.from_pretrained('distilgpt2')
    tokenize_dataset_field(boolq_dataset, gpt2_tokenizer)


    print(boolq_dataset['train'][0])
if __name__ == "__main__":
    main()

