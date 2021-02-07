import torch
import numpy as np
import pandas as pd


def try_gpu(i=0):
    if torch.cuda.device_count() >= i:
        return torch.cuda.device(f'cuda:{i}')
    else:
        return torch.device('cpu')

class Accumalator:
    """
    Accumalating sums over n variables
    """
    def __init__(self,n):
        self.data = [0.0] * n
    def add(self, *args ):
        self.data = [a+b for a,b in  zip(self.data,args)]
    def reset(self):
        self = [0.0] * len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]

class classification_model():
    def __init__(self,net, device = try_gpu()):
        self.net = net
        self.device = device
    def fit(self, loss_fn, optim, num_epochs,lr,train_iter):
        return 0
