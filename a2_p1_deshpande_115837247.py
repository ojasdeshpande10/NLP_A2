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




def main():
    boolq_dataset = load_dataset('google/boolq')
    emo_dataset = load_dataset('Blablablab/SOCKET', 'emobank#valence')
    print(boolq_dataset)
if __name__ == "__main__":
    main()

