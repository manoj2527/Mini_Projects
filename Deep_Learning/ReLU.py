import torch

class ReLU:

	def __init__(self):

		self.type = 1
		self.output = None
		self.gradInput = None

	def forward(self, input):

		self.output = input
		self.output[self.output < 0] = 0

		return self.output

	def backward(self, input, gradOutput):

		prop = self.output
		prop[prop >= 0] = 1

		self.gradInput = gradOutput*prop

		return self.gradInput

