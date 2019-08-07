import torch, math
import numpy as np

class RNN:

    def __init__(self,out_h,in_x,out_y,train):
        self.out_y = out_y
        self.in_x = in_x
        self.out_h = out_h 
        self.output = []
        self.train = train
        self.Whh = torch.randn(out_h,out_h).double()*math.sqrt(1/out_h)
        self.Wxh = torch.randn(out_h,in_x).double()*math.sqrt(1/in_x)
        self.Why = torch.randn(out_y,out_h).double()*math.sqrt(1/out_h)
        self.gradWhh = torch.zeros(out_h,out_h).double()
        self.gradWxh = torch.zeros(out_h,in_x).double()
        self.gradWhy = None
        self.pgWhh = torch.zeros(out_h,out_h).double()
        self.pgWxh = torch.zeros(out_h,in_x).double()
        self.pgWhy = torch.zeros(out_y,out_h).double()
        self.h = torch.zeros(out_h,1).double()
    
    def forward(self,input):
        self.h = np.tanh(torch.mm(self.Whh,self.h)+torch.mm(self.Wxh,input)) #outy*batchsize 
        y = torch.mm(self.Why,self.h)
        if self.train:
            self.output.append(self.h)

        return y

    def backward(self,grads,input):
        #grads - #batchsize*outy
        time = len(self.output)
        self.gradWhy = torch.mm(grads.t(),self.output[time-1].t())
        gradWhyNorm = torch.sum(torch.pow(self.gradWhy, 2))
        if(gradWhyNorm>1e2):
            self.gradWhy=self.gradWhy*(1e2/gradWhyNorm)
        delta = torch.mm(grads,self.Why).t()*(1-self.output[time-1]**2) # outh*batch
        for t in np.arange(time)[::-1]:
            if t!=0:
                self.gradWhh += torch.mm(delta, self.output[t-1].t())
                gradWhhNorm = torch.sum(torch.pow(self.gradWhh, 2))
                if (gradWhhNorm > 1e2):
                    self.gradWhh = self.gradWhh * (1e2 / gradWhhNorm)
            self.gradWxh[:,input[t]] += delta.reshape(-1)
            gradWxhNorm = torch.sum(torch.pow(self.gradWxh, 2))
            if (gradWxhNorm > 1e3):
                self.gradWxh = self.gradWxh * (1e2 / gradWxhNorm)
            if t!=0:
                delta = torch.mm(self.Whh.t(),delta)*(1-self.output[t-1]**2)
        return self.gradWhh,self.gradWxh,self.gradWhy
    
    def reset(self):
        self.gradWhh = torch.zeros(self.out_h,self.out_h).double()
        self.gradWxh = torch.zeros(self.out_h,self.in_x).double()
        self.gradWhy = None
        self.pgWhh = torch.zeros(self.out_h,self.out_h).double()
        self.pgWxh = torch.zeros(self.out_h,self.in_x).double()
        self.pgWhy = torch.zeros(self.out_y,self.out_h).double()
        self.h = torch.zeros(self.out_h,1).double()
        self.output = []

