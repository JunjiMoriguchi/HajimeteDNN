import numpy as np

hoge = np.array([0,1,2,3,4,5]) #numpyのarrayはListが入力
print(hoge)

hoge_2d = np.array([[0,1,2,],[3,4,5]]) # array
print(hoge_2d)

hoge_3d = np.array([[[0,1,2,],[3,4,5]],
                    [[6,7,8,],[9,10,11]]])
print(hoge_3d)
print(hoge_3d.shape, "total:", hoge_3d.size) #タプルで取得
print(hoge_3d.shape[0], hoge_3d.shape[1], hoge_3d.shape[2])

idx0,idx1,idx2, = hoge_3d.shape #idx情報の取得
print("copy",idx0,idx1,idx2)

#hoge_list = [[1,2],[3,4],[5,6]]
#print(len(hoge_list))
#print(len(np.array(hoge_list)))

print(hoge.shape)
print(hoge.reshape(3,-1).shape) #-1とすれば自動で変換

print(hoge_2d.reshape(-1)) #１次元に変換

crip = np.zeros([2,2,3]) # ch x W x H
crip[1,:,:] = hoge_2d # :は全てを選択
print(crip) 

print(hoge_2d)
print(np.sum(hoge_2d, axis=0))
print(np.sum(hoge_2d, axis=1))

print(np.arange(10))