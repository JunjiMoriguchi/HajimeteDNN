import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from img2col import im2col #モジュール読み込み
from col2img import col2im #モジュール読み込み
from nn import MiddleLayer,OutputLayer #モジュール読み込み

# hyper parameter
img_h = 8
img_w = 8
img_ch = 1
wb_width = 0.1
eta = 0.01
epoch = 50
batch_size = 8
interval = 10
n_sample = 200

class ConvLayer:
    def __init__(self,x_ch,x_h,x_w,n_flt,flt_h,flt_w,stride,pad):
        self.param = (x_ch,x_h,x_w,n_flt,flt_h,flt_w,stride,pad)
        self.w = wb_width * np.random.randn(n_flt,x_ch,flt_h,flt_w)
        self.b = wb_width * np.random.randn(1, n_flt) #self.b: 1,M
        self.y_ch = n_flt #self.y_ch; M
        self.y_h = (x_h - flt_h + 2*pad) // stride + 1 #self.y_h: Oh
        self.y_w = (x_w - flt_w + 2*pad) // stride + 1 #self.y_w: Ow
        # AdaGrad
        self.h_w = np.zeros((n_flt,x_ch,flt_h,flt_w)) + 1e-8 # 0割対策で1e-8を加算
        self.h_b = np.zeros((1,n_flt)) + 1e-8
    def forward(self, x): #x: B,C,Ih,Iw
        #初期化
        n_bt = x.shape[0]
        x_ch,x_h,x_w, n_flt,flt_h,flt_w, stride,pad = self.param
        y_ch = self.y_ch #M
        y_h = self.y_h #Oh
        y_w = self.y_w #Ow
        #im2col
        self.cols = im2col(x,flt_h,flt_w,y_h,y_w,stride,pad) # self.cols : CxFhxFw, BxOhxOw
        self.w_cols = self.w.reshape(n_flt, x_ch*flt_h*flt_w) # self.w_cols : M, CxFhxFw
        # 畳込み演算
        u = np.dot(self.w_cols, self.cols).T + self.b # u: M, BxOhxOw
        self.u = u.reshape(n_bt, y_h, y_w, y_ch).transpose(0,3,1,2) # self.u: B, M, Ow, Oh
        #活性化関数(ReLU)
        self.y = np.where(self.u <=0, 0, self.u) # self.y: B,M,Ow,Oh
    def backward(self, grad_y): # grad_y: B,M,Ow,Oh
        #初期化
        n_bt = grad_y.shape[0]
        x_ch,x_h,x_w, n_flt,flt_h,flt_w, stride,pad = self.param
        y_ch = self.y_ch #M
        y_h = self.y_h #Oh
        y_w = self.y_w #Ow
        # delta
        delta = grad_y * np.where(self.u <= 0, 0, 1) #delta:  B,M,Ow,Oh
        delta = delta.transpose(0,2,3,1).reshape(n_bt*y_h*y_w,y_ch) #delta: BxOwxOh,M
        #フィルタとバイアスの勾配
        grad_w = np.dot(self.cols, delta) #grad_w: CxFhxFw,M
        self.grad_w = grad_w.T.reshape(n_flt, x_ch, flt_h, flt_w) # self.grad_w: M,C,Fh,Fw
        self.grad_b = np.sum(delta, axis=0) 
        #入力の勾配
        grad_cols = np.dot(delta, self.w_cols) #grad_cols:BxOwxOh,CxFhxFw
        x_shape = (n_bt,x_ch,x_h,x_w)
        self.grad = col2im(grad_cols.T, x_shape, flt_h, flt_w, y_h, y_w, stride, pad) # self.grad: B,C,Ih,Iw
    def update(self,eta):
        self.h_w += self.grad_w * self.grad_w
        self.w -= eta / np.sqrt(self.h_w) * self.grad_w
        self.h_b += self.grad_b * self.grad_b
        self.b -= eta / np.sqrt(self.h_b) * self.grad_b

class PoolingLayer:
    def __init__(self, x_ch, x_h, x_w, pool, pad):
        self.params = (x_ch, x_h, x_w, pool, pad)
        self.y_ch = x_ch
        self.y_h = x_h//pool if x_h%pool == 0 else x_h//pool+1
        self.y_w = x_w//pool if x_w%pool == 0 else x_w//pool+1
    def forward(self, x):
        #初期化
        n_bt = x.shape[0]
        x_ch, x_h, x_w, pool, pad = self.params
        y_ch, y_h, y_w = self.y_ch, self.y_h, self.y_w
        #入力画像を行列に変換
        cols = im2col(x, pool, pool, y_h, y_w, pool, pad) # cols : CxPxP, BxOhxOw
        cols = cols.T.reshape(n_bt*y_h*y_w*x_ch, pool*pool) # cols : BxOhxOwC, PxP ->加算するためにCxPxPしない
        #max pooling
        y = np.max(cols, axis=1) #BOhOwC
        self.y = y.reshape(n_bt,y_h,y_w,x_ch).transpose(0,3,1,2)
        #最大値のidexを保持
        self.max_index = np.argmax(cols, axis=1)
    def backward(self,grad_y):#grad_y: B,C,Oh,Ow
        #初期化
        n_bt = grad_y.shape[0]
        x_ch,x_h,x_w,pool,pad = self.params
        y_ch, y_h, y_w = self.y_ch, self.y_h, self.y_w
        #出力の勾配の軸を入れ替え
        grada_y = grad_y.transpose(0,2,3,1) #BOwOhC
        #行列を作成
        grad_cols = np.zeros((pool*pool, grad_y.size)) #PP,BOwOhC
        #格配列の最大値であった要素にのみ出力の勾配を入れる
        grad_cols[self.max_index.reshape(-1),np.arange(grad_y.size)] = grad_y.reshape(-1)
        grad_cols = grad_cols.reshape(pool,pool,n_bt,y_h,y_w,y_ch) #PPBOhOwC
        grad_cols = grad_cols.transpose(5,0,1,2,3,4) #CPPBOhOw
        grad_cols = grad_cols.reshape(y_ch*pool*pool, n_bt*y_h*y_w) #CPP,BOhOw
        x_shape = (n_bt,x_ch,x_h,x_w)
        self.grad_x = col2im(grad_cols,x_shape,pool,pool,y_h,y_w,pool,pad) #BMOwOh


# input data sets
digits_data = datasets.load_digits()
input_data = digits_data.data
correct = digits_data.target
n_data = len(correct)
# 標準化
ave_input = np.average(input_data)
std_input = np.std(input_data)
input_data = (input_data - ave_input) / std_input
# one-hot
correct_data = np.zeros((n_data, 10)) # N, 10
for i in range(n_data):
    correct_data[i, correct[i]] = 1.0
# training & test data
index = np.arange(n_data)
index_train = index[index%3 !=0]
index_test = index[index%3 == 0]
input_train = input_data[index_train,:]
correct_train = correct_data[index_train,:]
input_test = input_data[index_test,:]
correct_test = correct_data[index_test,:]
n_train = input_train.shape[0]
n_test = input_test.shape[0]
# initialized layer
cl_1 = ConvLayer(img_ch, img_h, img_w, 10, 3, 3, 1, 1)
pl_1 = PoolingLayer(cl_1.y_ch, cl_1.y_h, cl_1.y_w, 2, 0)
n_fc_in = pl_1.y_ch * pl_1.y_h * pl_1.y_w #full conv in
ml_1 = MiddleLayer(n_fc_in, 100)
ol_1 = OutputLayer(100,10)

# Function forward
def forward_propagation(x):
    n_bt = x.shape[0]
    images = x.reshape(n_bt, img_ch, img_h, img_w)

    cl_1.forward(images)
    pl_1.forward(cl_1.y)
    fc_input = pl_1.y.reshape(n_bt, -1)
    ml_1.foward(fc_input)
    ol_1.foward(ml_1.y)

def backward_propagation(t):
    n_bt = t.shape[0]

    ol_1.backward(t)
    ml_1.backword(ol_1.grad_x)
    grad_img = ml_1.grad_x.reshape(n_bt,pl_1.y_ch,pl_1.y_w,pl_1.y_h)
    pl_1.backward(grad_img)
    cl_1.backward(pl_1.grad_x)

def update_wb():
    cl_1.update(eta)
    ml_1.update(eta)
    ol_1.update(eta)

def get_error(t, batch_size):
    return -np.sum(t * np.log(ol_1.y + 1e-7)) / batch_size

def forward_sample(inp, correct, n_sample):
    index_rand = np.arange(len(correct))
    np.random.shuffle(index_rand)
    index_rand = index_rand[:n_sample]
    x = inp[index_rand, :]
    t = correct[index_rand, :]
    forward_propagation(x)
    return x,t

# 誤差記録用
train_error_x = []
train_error_y = []
test_error_x = []
test_error_y = []

n_batch = n_train // batch_size
for i in range(epoch):

    # learning evaluation
    x,t = forward_sample(input_train, correct_train, n_sample)
    error_train = get_error(t, n_sample)

    x,t =forward_sample(input_test, correct_test,n_sample)
    error_test = get_error(t, n_sample)

    train_error_x.append(i)
    train_error_y.append(error_train)
    test_error_x.append(i)
    test_error_y.append(error_test)

    if i%interval == 0:
        print("Epoch:"+str(i)+"/"+str(epoch), 
                "Error_train:"+str(error_train),
                "Error_test:"+str(error_test))
    
    # learninng
    index_rand = np.arange(n_train)
    np.random.shuffle(index_rand)
    for j in range(n_batch):
        mb_index = index_rand[j*batch_size : (j+1)*batch_size]
        x = input_train[mb_index,:]
        t = correct_train[mb_index,:]
        forward_propagation(x)
        backward_propagation(t)
        update_wb()

# graph learning result
plt.plot(train_error_x,train_error_y,label="Train")
plt.plot(test_error_x,test_error_y,label="Test")
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Error")

plt.show()

# Estimation OK/NG
x,t = forward_sample(input_train, correct_train, n_train)
count_train = np.sum(np.argmax(ol_1.y, axis=1)==np.argmax(t,axis=1))
x,t = forward_sample(input_test,correct_test,n_test)
count_test = np.sum(np.argmax(ol_1.y, axis=1)==np.argmax(t,axis=1))

print("Acuuracy Train:", 
    str(count_train/n_train*100)+"%",
    "Accuracy Test:",
    str(count_test/n_test*100)+"%")
