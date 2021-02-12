import IAFlow as flw
import numpy as np
from time import time
import scipy.signal as sc
import core.cython_core as cc
import scipy


def conv_backward(dH, cache):
    '''
    The backward computation for a convolution function

    Arguments:
    dH -- gradient of the cost with respect to output of the conv layer (H), numpy array of shape (n_H, n_W) assuming channels = 1
    cache -- cache of values needed for the conv_backward(), output of conv_forward()

    Returns:
    dX -- gradient of the cost with respect to input of the conv layer (X), numpy array of shape (n_H_prev, n_W_prev) assuming channels = 1
    dW -- gradient of the cost with respect to the weights of the conv layer (W), numpy array of shape (f,f) assuming single filter
    '''

    # Retrieving information from the "cache"
    (X, W) = cache

    # Retrieving dimensions from X's shape
    (n_H_prev, n_W_prev) = X.shape

    # Retrieving dimensions from W's shape
    (f, f) = W.shape

    # Retrieving dimensions from dH's shape
    (n_H, n_W) = dH.shape

    # Initializing dX, dW with the correct shapes
    dX = np.zeros(X.shape)
    dW = np.zeros(W.shape)

    # Looping over vertical(h) and horizontal(w) axis of the output
    for h in range(n_H):
        for w in range(n_W):
            dX[h:h + f, w:w + f] += W * dH[h, w]
            dW += X[h:h + f, w:w + f] * dH[h, w]

    return dX, dW

#layer_1 = flw.maxPooling(2,2)
layer_1 = flw.convolutional_layer(h=3, w=3, func=flw.leakyReLU,
                                  filters=8, channels=3, stride=1)

python_time = []
cython_time = []

X = np.random.randn(2000, 28, 28, 3)


for i in range(3):
    '''star_time = time()
    Delta1 = layer_1.python_backward(x,delta)
    end_time = time()
    total_time = end_time - star_time
    print("The pooled with Python take %fs" % (total_time))
    python_time.append(total_time)
    print(Delta1.shape)'''


    star_time = time()
    layer_1.forwardpass(X)
    end_time = time()
    total_time = end_time-star_time
    print("The pooled with cython take %fs"%(total_time))
    cython_time.append(total_time)


#python_mean = sum(python_time)/len(python_time)
cython_mean = sum(cython_time)/len(cython_time)

#print('The mean time with python was %fs'%(python_mean))
print('The mean time with cython was %fs'%(cython_mean))
#print('The improvement was of %fx'%(python_mean/cython_mean))

