import numpy as np

# hyper parameter
wb_width = 0.1

class BaseLayer :
    def __init__(self, n_upper, n):
        self.w = wb_width * np.random.randn(n_upper, n)
        self.b = wb_width * np.random.randn(n)
        self.h_w = np.zeros((n_upper, n)) + 1e-8 # 0割対策で1e-8を加算
        self.h_b = np.zeros(n) + 1e-8
    def update(self, eta):
        # AdaGrad(重みとバイアスの更新量が学習が進むにつれて小さくなる)
        self.h_w += self.grad_w * self.grad_w
        self.w -= eta / np.sqrt(self.h_w) * self.grad_w
        self.h_b += self.grad_b * self.grad_b
        self.b -= eta / np.sqrt(self.h_b) * self.grad_b

class MiddleLayer(BaseLayer):
    def foward(self, x):
        self.x = x
        self.u = np.dot(x, self.w) + self.b
        self.y = np.where(self.u <= 0, 0, self.u) # ReLU
    def backword(self, grad_y):
        delta = grad_y * np.where(self.u <= 0, 0, 1) # grad ReLU
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)

class OutputLayer(BaseLayer):
    def foward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = np.exp(u) / np.sum(np.exp(u), axis=1).reshape(-1,1) # softmax
    def backward(self, t):
        delta = self.y - t
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)
