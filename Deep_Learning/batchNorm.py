import torch
##https://wiseodd.github.io/techblog/2016/07/04/batchnorm/

class batchNorm:

	def __init__(self):
		self.type = 4
		self.gamma = float(torch.rand(1))
		self.beta = float(torch.rand(1))
		self.inputNorm = None
		self.var = None
		self.mu = None
		self.dgamma = None
		self.dbeta = None
		
		self.pgG = torch.zeros(1).double()
		self.pgB = torch.zeros(1).double()

	def forward(self, input):
	    mu = input.mean(0)
	    var = input.var(0)

	    inputNorm = (input - mu) / torch.sqrt(var + 1e-8)
	    out = self.gamma * inputNorm + self.beta


	    self.inputNorm = inputNorm
	    self.var = var
	    self.mu = mu

	    return out

	def backward(self, input, gradOutput):

	    inputMu = input - self.mu
	    stdInv = 1.0/torch.sqrt(self.var + 1e-8)

	    gradInputNorm = gradOutput*self.gamma
	    dvar = torch.sum(gradInputNorm*inputMu, dim=0)*-0.5*stdInv**3
	    dmu = torch.sum(gradInputNorm*-stdInv, dim=0) + dvar*torch.mean(-2.0*inputMu, dim=0)

	    gradInput = (gradInputNorm*stdInv) + (dvar*2*inputMu/input.shape[0]) + (dmu/input.shape[0])
	    dgamma = torch.sum(gradOutput*self.inputNorm, dim=0)
	    dbeta = torch.sum(gradOutput, dim=0)

	    self.dgamma = dgamma
	    self.dbeta = dbeta

	    return gradInput