class Model:

	def __init__(self,malpha):

		self.Layers = []
		self.isTrain = None
		self.forwards = None
		self.test = False
		self.moment = malpha
	def addLayer(self, layer):
		self.Layers.append(layer)

	def forward(self, input):

		self.forwards = []
		output = input
		self.forwards.append(output)
		for i in range(len(self.Layers)):
			if (self.test) and self.Layers[i].type==6:
				pass
			else:
				output = self.Layers[i].forward(output)
				self.forwards.append(output)

		return output

	def backward(self, input, gradOutput):

		grads = gradOutput
		for i in range(len(self.Layers)-1,-1,-1):
			grads = self.Layers[i].backward(self.forwards[i], grads)


	def updateGradParam(self,moment,alpha):
		
		for i in range(len(self.Layers)):
			if (self.Layers[i].type == 0 or self.Layers[i].type == 2):
				self.Layers[i].pgW = moment*self.Layers[i].pgW -  alpha*self.Layers[i].gradW 
				self.Layers[i].W += self.Layers[i].pgW

				self.Layers[i].pgB = moment*self.Layers[i].pgB - alpha*self.Layers[i].gradB
				self.Layers[i].B += self.Layers[i].pgB

			if (self.Layers[i].type == 4):
				self.Layers[i].pgG = moment*self.Layers[i].pgG - alpha*self.Layers[i].dgamma  
				self.Layers[i].gamma += self.Layers[i].pgG  

				self.Layers[i].pgB = moment*self.Layers[i].pgB - alpha*self.Layers[i].dbeta 
				self.Layers[i].beta += self.Layers[i].pgB 

	def clearGradParam(self):

		self.forwards = []
		for i in range(len(self.Layers)):
			if (self.Layers[i].type == 0 or self.Layers[i].type == 2):
				self.Layers[i].gradW = 0
				self.Layers[i].gradB = 0

			if (self.Layers[i].type == 4):
				self.Layers[i].dgamma = 0
				self.Layers[i].dbeta = 0

	def rtest(self):
		self.test = not(self.test)