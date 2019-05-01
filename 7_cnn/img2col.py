import numpy as np

def im2col(image, flt_h, flt_w, out_h, out_w, stride, pad):
    n_bt,n_ch,img_h,img_w = image.shape #arrayのサイズを取得
    img_pad = np.pad(image, [(0,0),(0,0),(pad,pad),(pad,pad)], "constant") #padding

    cols = np.zeros((n_bt,n_ch,flt_h,flt_w,out_h,out_w))# 初期化
    for h in range(flt_h) :
        h_lim = h + stride*out_h
        for w in range(flt_w):
            w_lim = w + stride*out_w
            #バッチxRGBの組み合わせフィルタサイズを切り出して格納
            cols[:,:,h,w,:,:] = img_pad[:,:,h:h_lim:stride, w:w_lim:stride] #batch x ch x imgH x imgW
    cols = cols.transpose(1,2,3,0,4,5).reshape(n_ch*flt_h*flt_w, n_bt*out_h*out_w)

    return cols

img = np.array([[
                [ #R(HxW)
                    [1,2,3,4],
                    [5,6,7,8],
                    [9,10,11,12],
                    [13,14,15,16]
                ],
                [ #G(HxW)
                    [1,2,3,4],
                    [5,6,7,8],
                    [9,10,11,12],
                    [13,14,15,16]
                ],
                [ #B(HxW)
                    [1,2,3,4],
                    [5,6,7,8],
                    [9,10,11,12],
                    [13,14,15,16]
                ]
                ]
                ])
 #print(img.shape)
cols = im2col(img, 2, 2, 3, 3, 1, 0)
#print(cols)
#print(cols.shape)

