import torch, math
import functools as fp

class Convolution:

	def __init__(self, in_channels, filter_size, numfilters, stride):

		self.type = 2
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride
		self.out_depth = numfilters
		
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)
		
		self.W = torch.rand(self.out_depth, self.in_depth, self.filter_row, self.filter_col).double()
		self.W = self.W*math.sqrt(2/(self.in_depth*self.filter_row*self.filter_col))
		
		self.B = torch.rand(self.out_depth).double()*math.sqrt(2/(self.in_depth*self.filter_row*self.filter_col))
		
		self.output = None
		self.gradW = None
		self.gradB = None
		self.gradInput = None
		
		self.pgW = torch.zeros(self.out_depth, self.in_depth, self.filter_row, self.filter_col).double()
		self.pgB = torch.zeros(self.out_depth).double()

	def forward(self,input):
		s = self.stride
		sr = self.filter_row
		sc = self.filter_col
		ind = input.shape[1]
		n = input.shape[0]
		outd = self.out_depth
		outr = self.out_row
		outc = self.out_col 
		Out = torch.zeros((n,outd,outr,outc)).double()
		store = torch.zeros((n,outd,outr,outc)).double()

		for i in range(outr):
			for j in range(outc):
				inp = input[:,:,i*s:i*s+sr,j*s:j*s+sc]
				for d in range(outd):
					w = self.W[d,:,:,:]
					w = w.repeat(n,1,1,1)
					store[:,d,i,j] = torch.sum(w*inp,dim=3).sum(dim=2).sum(dim=1)
					Out[:,d,i,j] = store[:,d,i,j]+self.B[d]

		return Out

	def backward(self,activation_prev,delta):
		s = self.stride
		sr = self.filter_row
		sc = self.filter_col
		ind = activation_prev.shape[1]
		outd = self.out_depth
		outr = self.out_row
		outc = self.out_col 
		dw = torch.zeros(self.W.shape).double()
		db = torch.zeros(self.B.shape).double()

		n = activation_prev.shape[0]

		prev_delta = torch.zeros((n,ind,self.in_row,self.in_col)).double()


		for i in range(outr):
			for j in range(outc):
				inp = activation_prev[:,:,i*s:i*s+sr,j*s:j*s+sc]
				for d in range(outd):
					param_delta = delta[:,d,i,j]
					db[d] += float(param_delta.sum())
					pd = param_delta.unsqueeze(1).repeat(1,ind)
					pd = pd.unsqueeze(2).repeat(1,1,sr)
					pd = pd.unsqueeze(3).repeat(1,1,1,sc)
					dw[d,:,:,:] += torch.sum(pd*inp,dim=0)
					npw = self.W[d,:,:,:].repeat(n,1,1,1).double()
					prev_delta[:,:,i*s:i*s+sr,j*s:j*s+sc] += pd*npw

		
		self.gradB = db
		self.gradW = dw
		return prev_delta



