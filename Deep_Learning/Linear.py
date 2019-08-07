import torch, math

class Linear:

	def __init__(self, no_in_neurons, no_output_neurons):

		self.W = torch.rand(no_in_neurons, no_output_neurons).double()*math.sqrt(2/no_in_neurons)
		self.B = torch.rand(no_output_neurons, 1).double()*math.sqrt(2/no_in_neurons)

		self.type = 0
		self.output = None
		self.gradW = None
		self.gradB = None
		self.gradInput = None
		self.pgW = torch.zeros(no_in_neurons, no_output_neurons).double()
		self.pgB = torch.zeros(no_output_neurons, 1).double() 

		self.an = None
	def forward(self, input):

		self.output = torch.mm(input, self.W) + self.B.t()
		return self.output

	def backward(self, input, gradOutput):

		self.gradB = torch.sum(gradOutput, dim = 0).unsqueeze(1)
		self.gradW = torch.mm(input.t(), gradOutput)
		self.gradInput = torch.mm(gradOutput, self.W.t())

		return self.gradInput
