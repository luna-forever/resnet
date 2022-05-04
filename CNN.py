import cnn_utils
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

def zero_pad(X,pad):
    X_pad=np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),'constant',constant_values=0)
    return X_pad

def conv_single_step(a_pre_slice,W,b):
    s=np.multiply(a_pre_slice,W)
    Z=np.sum(s)+b
    return Z

np.random.seed(1)

#这里切片大小和过滤器大小相同
a_slice_prev = np.random.randn(4,4,3)
W = np.random.randn(4,4,3)
b = np.random.randn(1,1,1)

Z = conv_single_step(a_slice_prev,W,b)

print("Z = " + str(Z))

def conv_forward(A_prev,W,b,hparameters):
    (m,n_h_prev,n_W_prev,n_c_prev)=A_prev.shape
    (f,f,n_c_prev,n_c)=W.shape

    stride=hparameters["stride"]
    pad=hparameters["pad"]
    n_h=int((n_h_prev-f+2*pad)/stride)+1
    n_w = int((n_h_prev - f + 2 * pad) / stride) + 1
    Z=np.zeros((m,n_h,n_w,n_c))
    A_prev_pad=zero_pad(A_prev,pad)

    for i in range(m):
        a_prev_pad=A_prev_pad[i]
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):
                    vert_start=h*stride
                    vert_end=vert_start+f
                    hori_start=w*stride
                    hori_end=hori_start+f
                    a_slice_prev=a_prev_pad[vert_start:vert_end,hori_start:hori_end,:]
                    Z[i,h,w,c]=conv_single_step(a_slice_prev,W[:,:,:,c],b[0,0,0,c])

    cache=(A_prev,W,b,hparameters)
    return (Z,cache)

def pool_forward(A_prev,hparameters,mode='max'):
    (m,n_h_prev,n_W_prev,n_c_prev)=A_prev.shape

    stride=hparameters["stride"]
    f=hparameters["f"]
    n_h=int((n_h_prev-f)/stride)+1
    n_w = int((n_h_prev - f) / stride) + 1
    n_c=n_c_prev
    A=np.zeros((m,n_h,n_w,n_c))

    for i in range(m):
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):
                    vert_start=h*stride
                    vert_end=vert_start+f
                    hori_start=w*stride
                    hori_end=hori_start+f
                    a_slice_prev=A_prev[i,vert_start:vert_end,hori_start:hori_end,c]

                    if mode=='max':
                        A[i,h,w,c]=np.max(a_slice_prev)
                    elif mode=='average':
                        A[i,h,w,c]=np.mean(a_slice_prev)

    cache=(A_prev,hparameters)
    return (A,cache)

def conv_backward(dZ,cache):
    (A_prev,W,b,hparameters)=cache
    (m,n_h_prev,n_w_prev,n_c_prev)=A_prev.shape
    (m,n_h,n_w,n_c)=dZ.shape
    (f,f,n_c_prev,n_c)=W.shape
    pad=hparameters["pad"]
    stride=hparameters["stride"]

    dA_prev=np.zeros((m,n_h_prev,n_w_prev,n_c_prev))
    dW=np.zeros((f,f,n_c_prev,n_c))
    db=np.zeros((1,1,1,n_c))
    A_prev_pad=zero_pad(A_prev,pad)
    dA_prev_pad=zero_pad(dA_prev,pad)

    for i in range(m):
        a_prev_pad=A_prev_pad[i]
        da_prev_pad=dA_prev_pad[i]
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    hori_start = w * stride
                    hori_end = hori_start + f
                    a_slice=a_prev_pad[vert_start:vert_end,hori_start:hori_end,:]
                    da_prev_pad[vert_start:vert_end,hori_start:hori_end,:]+=W[:,:,:,c]*dZ[i,h,w,c]
                    dW[:,:,:,c]+=a_slice*dZ[i,h,w,c]
                    db[:,:,:,c]+=dZ[i,h,w,c]
        dA_prev[i,:,:,:]=da_prev_pad[pad:-pad,pad:-pad,:]

    return (dA_prev,dW,db)

def create_mask_from_window(x):
    mask=x==np.max(x)
    return mask

def distribute_value(dz,shape):
    (n_h,n_w)=shape
    average=dz/(n_h*n_w)
    a=np.ones(shape)*average
    return a

def pool_backward(dA,cache,mode='max'):
    (A_prev,hparameters) = cache
    f = hparameters["f"]
    stride = hparameters["stride"]

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        a_prev = A_prev[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    hori_start = w * stride
                    hori_end = hori_start + f
                    if mode=='max':
                        a_prev_slice=a_prev[vert_start:vert_end,hori_start:hori_end,c]
                        mask=create_mask_from_window(a_prev_slice)
                        dA_prev[i,vert_start:vert_end,hori_start:hori_end,c]+=np.multiply(mask,dA[i,h,w,c])
                    elif mode=='average':
                        da=dA[i,h,w,c]
                        shape=(f,f)
                        dA_prev[i,vert_start:vert_end,hori_start:hori_end,c]+=distribute_value(da,shape)

    return dA_prev

np.random.seed(1)
A_prev = np.random.randn(5, 5, 3, 2)
hparameters = {"stride" : 1, "f": 2}
A, cache = pool_forward(A_prev, hparameters)
dA = np.random.randn(5, 4, 2, 2)

dA_prev = pool_backward(dA, cache, mode = "max")
print("mode = max")
print('mean of dA = ', np.mean(dA))
print('dA_prev[1,1] = ', dA_prev[1,1])
print()
dA_prev = pool_backward(dA, cache, mode = "average")
print("mode = average")
print('mean of dA = ', np.mean(dA))
print('dA_prev[1,1] = ', dA_prev[1,1])




