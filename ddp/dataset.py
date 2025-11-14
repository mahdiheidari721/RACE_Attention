from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import random
import numpy as np
import numpy
import torch
import time
import math
from datasets import DatasetDict, concatenate_datasets
from tqdm.auto import tqdm



import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
# import tiktoken
import itertools
import matplotlib.pyplot as plt
from torch.profiler import profile, ProfilerActivity, record_function
from torch.profiler import schedule
import argparse

class YahooDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i], int(self.labels[i])