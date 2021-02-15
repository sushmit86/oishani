def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
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
    
    def fit(self, optimizer_fn,loss_fn,num_epochs,lr,train_iter,verbose = True):
        self.net.apply(self.init_weights)
        print('training on', self.device)
        self.net.to(self.device)
        num_batches = len(train_iter)
        optimizer = optimizer_fn(self.net.parameters(), lr=lr)
        time_array = np.zeros(num_epochs)
        acc_array = np.zeros(num_epochs)
        loss_array = np.zeros(num_epochs)
        for epoch in range(num_epochs):
            start_time = time.time()
            metric = Accumalator(3)
            self.net.train()
            for i,_train in enumerate(train_iter):
                X,y = _train[:,1:],_train[:,0].type(torch.long)
                optimizer.zero_grad()
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.net(X)
                l = loss_fn(y_hat, y)
                l.backward()
                optimizer.step()
                with torch.no_grad():
                    metric.add(l * X.shape[0], self.accuracy(y_hat, y), X.shape[0])
            train_l = metric[0]/metric[2]
            train_acc = metric[1] / metric[2]
            if (epoch%10 == 0 or epoch == (num_epochs - 1)) and verbose:
                print('epoch number', epoch)
                print(f'Training loss {train_l}')
                print(f'Training accuracy {train_acc}')
            end_time = time.time()
            time_array[epoch] = end_time - start_time
            acc_array[epoch] = train_acc
            loss_array[epoch] = train_l
        return (time_array,acc_array,loss_array)
               
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
                pred = np.concatenate((pred,torch.argmax(self.net(_data),axis = 1).numpy()))
        return pred