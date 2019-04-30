import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# get learinng data
iris_data = datasets.load_iris()
input_data = iris_data.data
correct = iris_data.target
n_data = len(correct)

#print(input_data.shape)
#print(correct.shape)

# standardization
ave_input = np.average(input_data, axis=0)
std_input = np.std(input_data, axis=0)
input_data = (input_data - ave_input) / std_input

# one-hot
correct_data = np.zeros((n_data,3))
for i in range(n_data):
    correct_data[i, correct[i]] = 1.0

# input/trainning
index = np.arange(n_data)
#print(index)
index_train = index[index%2 == 0]
index_test = index[index%2 != 0]
#print(index_train)

input_train = input_data[index_train, :]
correct_train = correct_data[index_train, :]
input_test = input_data[index_test, :]
correct_test = correct_data[index_test, :]

n_train = input_train.shape[0]
n_test = input_test.shape[0]

n_in = 4
n_mid = 50 #dropout時はニューロン２倍
n_out = 3

wb_width = 0.1
eta = 0.01
epoch = 1000
batch_size = 8
interval = 100

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
        self.y = np.exp(u) / np.sum(np.exp(u), axis=1, keepdims=True) # softmax
    def backward(self, t):
        delta = self.y - t
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)

class Dropout:
    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio
    def forward(self, x, is_train):
        if is_train:
            rand = np.random.rand(*x.shape)
            self.dropout = np.where( rand > self.dropout_ratio, 1, 0)
            self.y = x * self.dropout
        else :
            self.y = (1-self.dropout_ratio)*x
    def backward(self, grad_y):
        self.grad_x = grad_y * self.dropout

middle_layer_1 = MiddleLayer(n_in, n_mid)
dropout_1 = Dropout(0.5)
middle_layer_2 = MiddleLayer(n_mid, n_mid)
dropout_2 = Dropout(0.5)
output_layer = OutputLayer(n_mid, n_out)

def forward_propagation(x, is_train):
    middle_layer_1.foward(x)
    dropout_1.forward(middle_layer_1.y, is_train) # dropout
    middle_layer_2.foward(dropout_1.y)
    dropout_2.forward(middle_layer_2.y, is_train) # dropout
    output_layer.foward(dropout_2.y)
def backpropagation(t):
    output_layer.backward(t)
    dropout_2.backward(output_layer.grad_x)
    middle_layer_2.backword(dropout_2.grad_x)
    dropout_1.backward(middle_layer_2.grad_x)
    middle_layer_1.backword(dropout_1.grad_x)
def uppdate_wb():
    middle_layer_1.update(eta)
    middle_layer_2.update(eta)
    output_layer.update(eta)
def get_error(t, batch_size):
    return -np.sum(t * np.log(output_layer.y + 1e-7)) / batch_size #cross-emtropy error

train_error_x = []
train_error_y = []
test_error_x = []
test_error_y = []

n_batch = n_train // batch_size #切り捨て除算

for i in range(epoch):

    # test evalation
    forward_propagation(input_train, False)
    error_train = get_error(correct_train, n_train)
    forward_propagation(input_test, False)
    error_test = get_error(correct_test, n_test)
    
    test_error_x.append(i)
    test_error_y.append(error_test)
    train_error_x.append(i)
    train_error_y.append(error_train)

    if i%interval == 0:
        print("Epoch:"+str(i)+"/"+str(epoch),
                "Error_train:"+str(error_train),
                "Error_test:"+str(error_test))
    
    # batch learning
    index_random = np.arange(n_train)
    np.random.shuffle(index_random)
    for j in range(n_batch):

        mb_index = index_random[j*batch_size : (j+1)*batch_size]
        x = input_train[mb_index, :]
        t = correct_train[mb_index, :]

        forward_propagation(x,True)
        backpropagation(t)
        uppdate_wb()

# result
plt.plot(train_error_x, train_error_y, label="Train")
plt.plot(test_error_x, test_error_y, label="Test")
plt.legend() #凡例表示

plt.xlabel("Epochs")
plt.ylabel("Error")

plt.show()

forward_propagation(input_train, False)
count_train = np.sum(np.argmax(output_layer.y, axis=1) == np.argmax(correct_train, axis=1))

forward_propagation(input_test,False)
count_test = np.sum(np.argmax(output_layer.y, axis=1) == np.argmax(correct_test, axis=1))

print("Accuracy Train:", str(count_train/n_train*100) + "%", "Accuracy Test:", str(count_test/n_test*100)+"%")