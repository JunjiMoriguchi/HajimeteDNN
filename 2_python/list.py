import numpy as np

hoge = np.array([0,1,2,3,4,5]) #numpyのarrayはListが入力
print(hoge)

hoge_2d = np.array([[0,1,2,],[3,4,5]])
print(hoge_2d)

hoge_3d = np.array([[[0,1,2,],[3,4,5]],
                    [[6,7,8,],[9,10,11]]])
print(hoge_3d)
print(hoge_3d.shape, "total:", hoge_3d.size) #タプルで取得
print(hoge_3d.shape[0], hoge_3d.shape[1], hoge_3d.shape[2])

#hoge_list = [[1,2],[3,4],[5,6]]
#print(len(hoge_list))
#print(len(np.array(hoge_list)))