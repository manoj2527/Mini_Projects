import torch

class sigactiv:


	def __init__(self):

		self.type = 5
		self.output = None
		self.gradInput = None
		self.input = None

	def forward(self, input):

		self.input = input
		self.output = sigmoid(input)
		return self.output

	def backward(self, input, gradOutput):

		self.gradInput = gradOutput*self.output*(1-self.output)
		return self.gradInput

def sigmoid(x):
	return (1.0)/(1.0+torch.exp(-x))

