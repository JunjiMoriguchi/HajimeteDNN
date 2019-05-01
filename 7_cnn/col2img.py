import numpy as np

def col2im(cols, img_shape, flt_h, flt_w, out_h, out_w, stride, pad):
    n_bt, n_ch, img_h, img_w = img_shape
    #cols.shape : CFhFw, BOhOw
    cols = cols.reshape(n_ch, flt_h, flt_w, n_bt, out_h, out_w).transpose(3,0,1,2,4,5) # BatchxCHxFLT_HxFLT_WxOUT_HxOUT_W
    
    # pad + img_h + pad + stride - 1(画像がストライドで割り切れない場合を考慮)
    # pad + img_w + pad + stride - 1
    images = np.zeros((n_bt,n_ch,img_h+2*pad+stride-1,img_w+2*pad+stride-1))

    for h in range(flt_h): #filter height
        h_lim = h + stride*out_h
        for w in range(flt_w): #filter width
            w_lim = w + stride * out_w
            images[:,:,h:h_lim:stride,w:w_lim:stride] += cols[:,:,h,w,:,:]

    return images[:,:,pad:img_h+pad,pad:img_w+pad]

cols = np.ones((4,4))
img_shape = (1,1,3,3) # B,C,H,W
images = col2im(cols,img_shape,2,2,2,2,1,0) #IN,OUT,Fh,Fw,Oh,Ow,Stride,Padding
#print(images)
