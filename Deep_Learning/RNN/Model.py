import torch,math
import numpy as np

class Model:
    def __init__(self,malpha):
        self.layer = []
        self.train = None
        self.moment = malpha
    
    def addLayer(self,layer):
        self.layer.append(layer)
    
    def forward(self,input):
        p=self.layer[0].forward(input)
        return p

    def backward(self,gradoutput,input):
        return self.layer[0].backward(gradoutput,input)
    
    # def updategrad(self,moment,alpha):
    #     self.layer[0].pgWhh = moment*self.layer[0].pgWhh - alpha*self.layer[0].gradWhh
    #     self.layer[0].Whh += self.layer[0].pgWhh
    #     self.layer[0].pgWxh = moment*self.layer[0].pgWxh - alpha*self.layer[0].gradWxh
    #     self.layer[0].Wxh += self.layer[0].pgWxh
    #     self.layer[0].pgWhy = moment*self.layer[0].pgWhy - alpha*self.layer[0].gradWhy
    #     self.layer[0].Why += self.layer[0].pgWhy

    def updategrad(self,moment,alpha):
        self.layer[0].Whh += -alpha*self.layer[0].gradWhh
        self.layer[0].Wxh += -alpha*self.layer[0].gradWxh
        self.layer[0].Why += -alpha*self.layer[0].gradWhy

    def cleargrad(self):
        self.layer[0].reset()
    
    def test(self):
        self.layer[0].train = not(self.layer[0].train)
