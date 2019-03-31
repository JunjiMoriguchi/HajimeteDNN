import matplotlib.pyplot as plt
import numpy as np

X = np.arange(-1.0, 1.1, 0.1)
Y = np.arange(-1.0, 1.1, 0.1)

input_data = []
correct_data = []

for x in X :
    for y in Y:
        input_data.append([x,y])
        if y < np.sin(np.pi * x):
            correct_data.append([0,1]) # 下の領域
        else:
            correct_data.append([1,0]) # 上の領域

n_data = len(correct_data)
input_data = np.array(input_data) # (2,441)
correct_data = np.array(correct_data) # (2,441)


# number of neuron
n_in = 2
n_mid = 6
n_out = 2

# hyper prameters
wb_width = 0.01
eta      = 0.1
epoch    = 101
interval = 10

class MiddleLayer:
    def __init__(self, n_upper, n):
        # n_upper : number of upper neuron
        # n       : number of self neuron
        self.w = wb_width * np.random.randn(n_upper, n)
        self.b = wb_width * np.random.randn(n)
        pass
    def forward(self, x):
        self.x = x
        u = np.dot(x,self.w) + self.b
        self.y = 1 / (1 + np.exp(-u)) #sigmoid
        pass
    def backward(self, grad_y):
        delta = grad_y * (1 - self.y)*self.y
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)
        pass
    def update(self, eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b
        pass
    pass

class OutputLayer:
    def __init__(self, n_upper, n):
        # n_upper : number of upper neuron
        # n       : number of self neuron
        self.w = wb_width * np.random.randn(n_upper, n)
        self.b = wb_width * np.random.randn(n)
        pass
    def forward(self, x):
        self.x = x
        u = np.dot(x,self.w) + self.b
        self.y = np.exp(u) / np.sum(np.exp(u), axis=1, keepdims=True)
        pass
    def backward(self, t):
        delta = self.y - t
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)
        pass
    def update(self, eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b
        pass
    pass


if __name__ == "__main__":

    middle_layer = MiddleLayer(n_in, n_mid)
    output_layer = OutputLayer(n_mid, n_out)

    sin_data = np.sin(np.pi*X) # validation data

    for i in range(epoch):
        index_random = np.arange(n_data)
        np.random.shuffle(index_random)

        total_error = 0
        x_1 = []
        y_1 = []
        x_2 = []
        y_2 = []
        
        for idx in index_random:
            x = input_data[idx] # (2,1)
            t = correct_data[idx] # (2, 1)

            middle_layer.forward(x.reshape(1,2)) #(1,2)
            output_layer.forward(middle_layer.y)

            output_layer.backward(t.reshape(1,2))
            middle_layer.backward(output_layer.grad_x)

            middle_layer.update(eta)            
            output_layer.update(eta)

            if i%interval == 0:
                y = output_layer.y.reshape(-1)
                total_error += - np.sum(t * np.log(y + 1e-7))

                # classfication
                if (y[0] > y[1]):
                    x_1.append(x[0])
                    y_1.append(x[1])
                else:
                    x_2.append(x[0])
                    y_2.append(x[1])

        if i%interval == 0:
            plt.plot(X, sin_data, linestyle="dashed")
            plt.scatter(x_1,y_1,marker="+")
            plt.scatter(x_2,y_2,marker="x")
            #plt.show()
            plt.pause(0.01)

    pass
