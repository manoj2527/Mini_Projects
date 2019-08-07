import torch

class Dropout:
    def __init__(self,p):
        self.type = 6
        self.an = None
        self.p  = p
    def forward(self, X):
        
        self.an =  torch.bernoulli(torch.ones([1,X.shape[1]])*self.p).double()
        return X*self.an

    def backward(self, activation_prev, delta):
        return delta*self.an
