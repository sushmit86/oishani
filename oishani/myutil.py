import torch
import numpy as np
import pandas as pd


def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.cuda.device(f'cuda:{i}')
    else:
        return torch.device('cpu')

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

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
        '''
        net: Pytorch neural network model
        '''
        self.net = net
        self.device = device
    
    def fit(self, loss_fn,num_epochs,lr,train_iter):
        self.net.apply(self.init_weights)
        print('training on', self.device)
        self.net.to(self.device)
        num_batches = len(train_iter)
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        for epoch in range(num_epochs):
            print(epoch)
            metric = Accumalator(3)
            self.net.train()
            for i,_train in enumerate(train_iter):
                X,y = _train[:,1:],_train[:,0]
                optimizer.zero_grad()
                X, y = X.to(self.device), y.to(self.device)
                y = torch.tensor(y, dtype=torch.long, device=self.device)
                y_hat = self.net(X)
                l = loss_fn(y_hat, y)
                l.backward()
                optimizer.step()
                with torch.no_grad():
                    metric.add(l * X.shape[0], self.accuracy(y_hat, y), X.shape[0])
            train_l = metric[0]/metric[2]
            train_acc = metric[1] / metric[2]
            if epoch%5 == 0 or epoch == (num_epochs - 1):
                print('epoch number', epoch)
                print(f'Training loss {train_l}')
                print(f'Training accuracy {train_acc}')
                print(metric[1])

    def init_weights(self,m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
            
    def accuracy(self,y_hat,y):
        y_pred = torch.argmax(y_hat,axis=1)
        y_pred = y_pred.type(y.dtype)
        return float(torch.sum(y_pred == y))
    
    def prediction(self,data_iter):
        if isinstance(self.net, torch.nn.Module):
            self.net.eval() 
        for i,_data in enumerate(data_iter):
            if i == 0:
                pred = np.zeros(_data.shape[0])
                pred = torch.argmax(self.net(_data),axis = 1).numpy()
            else:
                pred[:] = np.concatenate((torch.argmax(self.net(_data),axis = 1).numpy()),pred)
        return pred