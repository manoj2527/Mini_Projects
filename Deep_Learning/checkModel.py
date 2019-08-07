import argparse, torch, torchfile
from Model import Model
from Linear import Linear
from ReLU import ReLU

def buildModel(config):
	f = open(config, "r")
	n = int(f.readline())

	testModel = Model()

	for i in range(n):
		tokens = f.readline().split()
		if tokens[0] == "linear":
			inpNodes = int(tokens[1])
			outNodes = int(tokens[2])
			
			linLayer = Linear(inpNodes, outNodes)
			testModel.addLayer(linLayer)

		if tokens[0] == "relu":
			reluLayer = ReLU()
			testModel.addLayer(reluLayer)


	tokens = f.readline().split()
	weightsPath = tokens[0]
	weights = torchfile.load(weightsPath)
	tokens = f.readline().split()
	biasPath = tokens[0]
	biases = torchfile.load(biasPath)


	cnt = 0
	for i in range(n):
		if testModel.Layers[i].type == 0:
			testModel.Layers[i].W = torch.from_numpy(weights[cnt].T).double()
			testModel.Layers[i].B = torch.from_numpy(biases[cnt].T).double().unsqueeze(1)
			cnt += 1

	f.close()
	return testModel

if __name__ == '__main__':

	parser = argparse.ArgumentParser("checkModel.py")
	parser.add_argument("-config", dest="config" , help="/path/to/modelConfig.txt", type=str)
	parser.add_argument("-i", dest="inp" , help="/path/to/input.bin", type=str)
	parser.add_argument("-og", dest="gradOut" , help="/path/to/gradOutput.bin", type=str)
	parser.add_argument("-o", dest="out" , help="/path/to/Output.bin", type=str)
	parser.add_argument("-ow", dest="gradW" , help="/path/to/gradWeight.bin", type=str)
	parser.add_argument("-ob", dest="gradB" , help="/path/to/gradB.bin", type=str)
	parser.add_argument("-ig", dest="gradInp" , help="/path/to/gradInput.bin", type=str)
	args = parser.parse_args()
	
	Inp = torchfile.load(args.inp)
	batchSize = Inp.shape[0]
	Inp = torch.from_numpy(Inp).double()
	Inp = Inp.view(batchSize,-1)

	gradOut = torchfile.load(args.gradOut)
	gradOut = torch.from_numpy(gradOut).double()

	testModel = buildModel(args.config)
	testOutput = testModel.forward(Inp)

	with open(args.out, 'wb') as f: 
		torch.save(testOutput, f)
		f.close()

	testOutput = testModel.backward(Inp, gradOut)

	
	cnt = 0
	testW = []
	testB = []
	for i in range(len(testModel.Layers)):
		if testModel.Layers[i].type == 0:
			testW.append(testModel.Layers[i].W.t())
			testB.append(testModel.Layers[i].B.squeeze())
			cnt += 1 
	
	with open(args.gradW, 'wb') as f: 
		torch.save(testW, f)
		f.close()

	with open(args.gradB, 'wb') as f: 
		torch.save(testB, f)
		f.close()


	gradInp = testModel.backward(Inp, gradOut)
	
	with open(args.gradInp, 'wb') as f: 
		torch.save(gradInp, f)
		f.close()
