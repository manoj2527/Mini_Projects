class Flatten:
    def __init__(self):
        self.type = 3
        self.in_batch = None
        self.r = None
        self.c = None
        self.k = None
    
    def forward(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.view(self.in_batch, self.r * self.c * self.k)

    def backward(self, activation_prev, delta):
        return delta.view(self.in_batch, self.r, self.c, self.k)
