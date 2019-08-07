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
import numpy as np
from Dropout import Dropout

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == '__main__':

	parser = argparse.ArgumentParser("trainModel.py")
	parser.add_argument("-modelName", dest="modelName" , help="Model Name", type=str)
	parser.add_argument("-data", dest="xtrain" , help="/path/to/train/data.bin", type=str)
	parser.add_argument("-target", dest="ytrain" , help="/path/to/target/labels.bin", type=str)
	parser.add_argument("-a", dest="a" , help="learning rate", type=float)
	parser.add_argument("-b", dest="b" , help="batch size", type=int)
	parser.add_argument("-e", dest="e" , help="epochs", type=int)
	parser.add_argument("-loadModel", dest="loadModel" , help="to load a model",default=False, type=bool)
	args = parser.parse_args()

	"""
	if not os.path.exists(args.modelName):
		os.makedirs(args.modelName)
	else:
		shutil.rmtree(args.modelName)
		os.makedirs(args.modelName)
	"""

	#Loading And Standardising The Data
	model = []
	Xtrain = torchfile.load(args.xtrain)
	Xtrain = torch.from_numpy(Xtrain).double()
	noTrain = Xtrain.shape[0]
	model.append(Xtrain.mean(0))
	Xtrain -= Xtrain.mean(0)
	model.append(Xtrain.std(0))
	Xtrain /= Xtrain.std(0)
	model.append(torch.max(Xtrain))
	Xtrain /= torch.max(Xtrain)
	Xtrain = Xtrain.view(noTrain,1,Xtrain.shape[1],Xtrain.shape[2])

	test =  torch.randperm(noTrain)
	Xtest = Xtrain[test[0:5000],:,:,:]
	Xtrain = Xtrain[test[5000:],:,:,:]

	Ytrain = torchfile.load(args.ytrain)
	Ytrain = torch.from_numpy(Ytrain).long().unsqueeze(1)

	Ytest = Ytrain[test[0:5000],:]
	Ytrain = Ytrain[test[5000:],:]

	noTrain = Xtrain.shape[0]




	batchSize = args.b
	epochs = args.e
	alpha = args.a
	moment = 0.9



	myModel = Model(moment)
	myModel.addLayer(Flatten())
	myModel.addLayer(Linear(108*108,80))
	myModel.addLayer(batchNorm())
	myModel.addLayer(sigactiv())
	myModel.addLayer(Dropout(0.7))
	myModel.addLayer(Linear(80,20))
	myModel.addLayer(batchNorm())
	myModel.addLayer(sigactiv())
	myModel.addLayer(Linear(20,10))
	myModel.addLayer(batchNorm())
	myModel.addLayer(sigactiv())
	myModel.addLayer(Linear(10,6))
	criterion = Criterion()



	if args.loadModel:
		model = torch.load("modelParams.txt")
		k = 3
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

	numBatches = math.ceil(float(noTrain)/batchSize)

	for epoch in range(epochs):

		shuffle = torch.randperm(noTrain)
		Xtrain = Xtrain[shuffle]
		Ytrain = Ytrain[shuffle]

		trainAcc = 0

		for batchNum in range(numBatches):
			Xbatch = Xtrain[batchNum*batchSize:(batchNum+1)*batchSize,:]
			Ybatch = Ytrain[batchNum*batchSize:(batchNum+1)*batchSize,:]
			output = myModel.forward(Xbatch)
			maxOutput = torch.max(output,1)
			maxOutput = maxOutput[1].unsqueeze(1).long()

			#print(maxOutput.t())
			#print(Ybatch.t())

			trainAcc += (float(torch.sum(Ybatch == maxOutput)))/batchSize*100

			loss = criterion.forward(output,Ybatch)
			gradLoss = criterion.backward(output,Ybatch)

			myModel.backward(Xbatch,gradLoss)
			myModel.updateGradParam(moment,alpha)
			myModel.clearGradParam()

			if batchNum%25 == 24: 
				print("{}/{} TrainAcc = {}".format(batchNum+1,numBatches,trainAcc/25))
				trainAcc = 0

		myModel.rtest()
		output = myModel.forward(Xtrain)
		myModel.rtest()
		maxOutput = torch.max(output,1)
		pred = maxOutput[1].unsqueeze(1).long()
		test = (pred==Ytrain)

		myModel.rtest()
		voutput = myModel.forward(Xtest)
		myModel.rtest()
		vmaxOutput = torch.max(voutput,1)
		vpred = vmaxOutput[1].unsqueeze(1).long()
		vtest = (vpred==Ytest)
		print("epoch = {}/{} TAccu = {} , VAccu = {}".format(epoch+1,epochs,float(test.sum())/noTrain*100,float(vtest.sum())/50))

		


for l in myModel.Layers:
	if l.type == 0 or l.type == 2: 
		model.append(l.W) 
		model.append(l.B)
	if l.type == 4:
		model.append(l.gamma)
		model.append(l.beta)

with open("modelParams.txt", 'wb') as f: 
		torch.save(model, f)
		f.close()
print("Model Saved... ")
		



