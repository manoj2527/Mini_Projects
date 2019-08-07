import argparse, torchfile, torch, os, shutil, math
from Model import Model
from Linear import Linear
from ReLU import ReLU
from Criterion import Criterion
from Convolution import Convolution
from Flatten import Flatten
from batchNorm import batchNorm
from sigmoid import sigactiv
import pandas as pd


Xtest = torchfile.load("Test/test.bin")
Xtest = torch.from_numpy(Xtest).double()

noTrain = Xtest.shape[0]

moment = 0.9




myModel = Model(moment)
myModel.addLayer(Flatten())
myModel.addLayer(Linear(108*108,80))
myModel.addLayer(batchNorm())
myModel.addLayer(sigactiv())
myModel.addLayer(Linear(80,20))
myModel.addLayer(batchNorm())
myModel.addLayer(sigactiv())
myModel.addLayer(Linear(20,10))
myModel.addLayer(batchNorm())
myModel.addLayer(sigactiv())
myModel.addLayer(Linear(10,6))
criterion = Criterion()





model = torch.load("modelParams.txt")
k = 3

Xtest -= model[0]
Xtest /= model[1]
Xtest /= model[2]
Xtest = Xtest.view(noTrain,1,Xtest.shape[1],Xtest.shape[2])


for l in myModel.Layers:
	if l.type == 0 or l.type == 2: 
		l.W = model[k]
		l.B = model[k+1]

		k+=2

	if l.type == 4:
		l.gamma = model[k]
		l.beta = model[k+1]	

		k+=2
print("Model Loaded... ")



output = myModel.forward(Xtest)
maxOutput = torch.max(output,1)
pred = maxOutput[1].unsqueeze(1).long().numpy()
pred = pd.DataFrame(pred)
pred.columns = ['label']
pred.index.name = 'id'
pred.to_csv("Test/test.csv",sep=',')

