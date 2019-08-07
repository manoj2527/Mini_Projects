import torch

class Criterion:

	def __init__(self):
		self.loss = None
		self.soft = None

	def forward(self, input, target):

		target = target.long()

		exp_inp = torch.exp(input)
		self.soft = exp_inp/(torch.sum(exp_inp,dim=1).unsqueeze(1))

		loss = -torch.log(self.soft.gather(1,target.view(-1,1)))
		avg_loss = torch.sum(loss)
		avg_loss = avg_loss/input.shape[0]

		self.loss = avg_loss
		return avg_loss

	def backward(self, input, target):
		target = target.long()
		onehotTarget = torch.zeros(input.shape)
		onehotTarget = onehotTarget.scatter(1,target,1).double()

		gradInput = self.soft
		gradInput -= onehotTarget

		gradInput = gradInput/input.shape[0]
		return gradInput