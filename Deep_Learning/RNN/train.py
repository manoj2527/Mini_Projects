import torch
import numpy as np
import math
from Model import Model
from RNN import RNN
from Criterion import Criterion
import pandas as pd
import argparse, torchfile, os, shutil
from copy import deepcopy
from tqdm import tqdm

def one_hot_encode(X, labels):
	X.shape = (X.shape[0], 1)
	newX = np.zeros((X.shape[0], len(labels)))
	label_encoding = {}
	for i, l in enumerate(labels):
		label_encoding[l] = i
	for i in range(X.shape[0]):
		newX[i, label_encoding[X[i,0]]] = 1
	return newX

f = open('train_data.txt')
input = []
encode = []
while True:
    line = f.readline()
    if line=="":
        break
    else:
        line = line.rstrip("\n").split()
        line = [int(x) for x in line]
        input.append(np.array(line))
        encode.extend(line)

f1 = open('test_data.txt')
test = []
while True:
    line = f1.readline()
    if line=="":
        break
    else:
        line = line.rstrip("\n").split()
        line = [int(x) for x in line]
        test.append(line)
        encode.extend(line)

ylabels = []
f2 = open("train_labels.txt")
while True:
    line = f2.readline()
    if line=="":
        break
    else:
        line = line.rstrip("\n")
        ylabels.append([int(line)])

encode = deepcopy(np.unique(encode))
encode.sort()
code = {}
for i in range(len(encode)):
    code[encode[i]] = i 
inputlen = len(encode)
encode_time = []
itarget = []
for x in input:
    ret = one_hot_encode(np.array(x),encode)
    encode_time.append(ret)
    temp = []
    for p in x:
        temp.append(code[p])
    itarget.append(temp)
test_time = []
for x in test:
    ret = one_hot_encode(np.array(x),encode)
    test_time.append(ret)
print("preprocess completed")

moment = 0
lr = 0.05
epochs = 10

myModel = Model(moment)
myModel.addLayer(RNN(2,inputlen,2,True))
criterion = Criterion()
lth = len(input)
for r in tqdm(range(epochs)):
    if r%5==0:
        lr=lr/3.0
    crct = 0
    for i in tqdm(range(lth)):
        ret = encode_time[i]
        out = None
        for j in range(len(ret)):
            out = myModel.forward(torch.tensor(ret[j].reshape((len(ret[j]),-1))))
        maxOutput = torch.max(out.t(),1)[1].unsqueeze(1).long()
        if maxOutput[0]==torch.tensor(ylabels[i]):
            crct+=1
        loss = criterion.forward(out.t(),torch.tensor(ylabels[i]).unsqueeze(1))
        gradloss = criterion.backward(out.t(),torch.tensor(ylabels[i]).unsqueeze(1))
        #print(loss,gradloss,maxOutput[0])
        #crct+=loss
        myModel.backward(gradloss,itarget[i])
        myModel.updategrad(moment,lr)
        myModel.cleargrad()
    tqdm.write("iter "+str(r)+" "+str((1.0*crct)/len(input)))

print("completed")

f3 = open("test_labels.txt","w+")
f3.write("id,label\n")
for i in range(len(test_time)):
    ret = test_time[i]
    out = None
    for j in range(len(ret)):
       out = myModel.forward(torch.tensor(ret[j].reshape((len(ret[j]),-1))))
    maxOutput = torch.max(out.t(),1)[1].unsqueeze(1).long()
    f3.write(str(i)+","+str(int(maxOutput[0][0]))+"\n")

f3.close()
f1.close()
f.close()
