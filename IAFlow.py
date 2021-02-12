import numpy as np
from matplotlib.pyplot import plot, show
from scipy import signal
import scipy as sc
import core.cython_core as cc
import core.cython_core_functions as cf
from time import time

'''Funciones de activación junto con sus derivadas parciales'''


def sigmoid(x):
    if len(x.shape) > 2:
        xArray = np.array(np.reshape(x, x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]))
        Y = cf.sigmoid(xArray)
        Y = np.reshape(Y, (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
    else:
        xArray = np.array(np.reshape(x, x.shape[0] * x.shape[1]))
        Y = cf.sigmoid(xArray)
        Y = np.reshape(Y, (x.shape[0], x.shape[1]))
    return np.array(Y)


def reluFunction(x):
    derivative = np.array(x)
    for i in np.nditer(derivative, op_flags=['readwrite', 'updateifcopy']):
        if i >= 0:
            i[...] = 1
        else:
            i[...] = 0
    return derivative


def leakyReLU_func(x, alpha):
    x = np.array(x)
    if len(x.shape) > 2:
        xArray = np.reshape(x, x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3])
        Y = cf.leakyReLU(xArray, alpha)
        Y = np.reshape(Y, (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
    else:
        xArray = np.reshape(x, x.shape[0] * x.shape[1])
        Y = cf.leakyReLU(xArray, alpha)
        Y = np.reshape(Y, (x.shape[0], x.shape[1]))
    return np.array(Y)


def leakyReLU_derivative(x, alpha):
    x = np.array(x)
    if len(x.shape) > 2:
        xArray = np.reshape(x, x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3])
        Y = cf.leakyReluDerivative(xArray, alpha)
        Y = np.reshape(Y, (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
    else:
        xArray = np.reshape(x, x.shape[0] * x.shape[1])
        Y = cf.leakyReluDerivative(xArray, alpha)
        Y = np.reshape(Y, (x.shape[0], x.shape[1]))
    return np.array(Y)


def softmaxAct(x):
    x = cf.softmax(x)
    return np.array(x)


softmax = (lambda x: softmaxAct(x),
           None)

sigm = (lambda x: sigmoid(x),
        lambda x: sigmoid(x) * (1 - sigmoid(x)))

relu = (lambda x: np.maximum(0, x),
        lambda x: reluFunction(x))

tanh = (lambda x: np.tanh(x),
        lambda x: 1 - np.tanh(x) ** 2)

leakyReLU = (lambda x: leakyReLU_func(x, 0.01),
             lambda x: leakyReLU_derivative(x, 0.01))

'''Funciones de coste'''


def softmaxCost(z, y):
    temp = -(y * np.log(z + 1e-20))
    J = np.sum(temp) / z.shape[0]
    return J


def classificationCost(z, y):
    temp = (1 - y) * np.log((1 - z) + 1e-20)
    temp = y * np.log(z + 1e-20) + temp
    temp = np.sum(temp)
    J = - (1 / len(y)) * temp
    return J


def classificationCostwReg(weights, z, y, regLambda):
    regLambda = regLambda
    temp = (1 - y) * np.log(1 - z[len(z) - 1])
    temp = y * np.log(z[len(z) - 1]) + temp
    temp = np.sum(temp)
    reg = 0
    for i in weights:
        tempW = i.W ** 2
        reg += np.sum(tempW)
    reg = (regLambda / (2 * len(y))) * reg
    J = - (1 / len(y)) * temp
    J += reg
    return J


def regressionCost(weights, z, y, regLambda):
    J = np.sum(np.power(z[len(z) - 1] - y, 2)) / len(y)
    reg = 0
    for i in weights:
        tempW = i.W ** 2
        reg += np.sum(tempW)
    reg = (regLambda / (2 * len(y))) * reg
    J += reg
    return J


def gradientClipping(x, threshold):
    original_shape = x.shape
    x = x.flatten()
    x = cc.gradientClipping(x, threshold)
    x = np.reshape(x, original_shape)
    return x


class snn:

    def __init__(self, size, func, actFunc, costFunc, regLambda):
        self.regLambda = regLambda
        self.costFunc = costFunc
        self.actFunc = actFunc
        self.costs = {
            "classificationwReg": classificationCostwReg,
            "classification": classificationCost,
            "regression": regressionCost
        }
        self.size = size
        self.func = func
        weights = []
        for i in range(len(size) - 1):
            weights.append(neural_layer(size[i], size[i + 1], func))
        self.weights = weights

    def loadParam(self, name):
        self.weights = np.load(name)

    def insertData(self, x, y):
        self.x = x
        self.Y = y

    def feedforward(self, x=None):
        if x is not None:
            self.x = x
        z = [self.x]
        for i in range(len(self.weights)):
            if i == len(self.weights) - 1:
                z.append(self.actFunc[0]((z[i] @ self.weights[i].W) + self.weights[i].b))
            else:
                z.append(self.func[0]((z[i] @ self.weights[i].W) + self.weights[i].b))
        self.z = z

    def backpropagation(self):
        self.Delta = []
        d = []
        dtemp = self.z[len(self.z) - 1] - self.Y
        d.insert(0, dtemp)
        for i in range(len(self.z) - 2, 0, -1):
            dtemp = (d[0] @ np.transpose(self.weights[i].W)) * self.func[1](self.z[i])
            d.insert(0, dtemp)
        for i in range(len(d)):
            self.Delta.append((1 / len(self.x)) * (np.transpose(self.z[i]) @ d[i]) + \
                              ((self.regLambda / len(self.x))) * self.weights[i].W)

    def gradientDescent(self, rate, batch):
        it = []
        cost = []
        for i in range(batch):
            self.feedforward()
            self.backpropagation()
            for j in range(len(self.weights)):
                self.weights[j].W = self.weights[j].W - (rate * self.Delta[j])
                self.weights[j].b = self.weights[j].b - (rate * np.mean(self.Delta[j], axis=0, keepdims=True))
            J = self.costs[self.costFunc](self.weights, self.z, self.Y, self.regLambda)
            print("The cost function in the iteration %i is %f" % (i + 1, J))
            it.append(i + 1)
            cost.append(J)
        plot(it, cost)
        show()

    def saveParam(self, fname):
        np.save(fname, self.weights)


'''----------------Layer classes---------------------'''


class maxPooling():

    def __init__(self, stride, size):
        self.stride = stride
        self.size = size

    def forwardpass(self, x):
        pooledArray, positions = cc.maxPoolForward(x, self.size, self.stride)
        self.pooledArray = np.array(pooledArray)
        self.positions = np.array(positions)

    def backwardpass(self, x, delta):
        backPool = cc.maxPoolingBack(x.shape, delta, self.positions)
        return np.array(backPool)


class convolutional_layer():

    def __init__(self, h, w, func, filters, channels, stride, input_layer=False):
        self.stride = stride
        self.func = func
        self.kernel = np.random.normal(0, .05, (h, w, channels, filters))
        self.bias = np.random.normal(0, .05, (filters))
        self.input_layer = input_layer

    def forwardpass(self, x):
        shape = ((x.shape[1] - self.kernel.shape[1]) / self.stride) + 1
        new_shape = (int(x.shape[0]), int(shape), int(shape), int(self.kernel.shape[3]))
        if shape.is_integer():
            convolved = cc.convolution(x, np.rot90(self.kernel, 2, axes=(0, 1)), self.stride, new_shape)
            convolved = np.array(convolved)
            for filter in range(self.bias.shape[0]):
                convolved[:, :, :, filter] += self.bias[filter]
            self.convolved = convolved
            self.activation = self.func[0](self.convolved)
        else:
            raise ValueError('Invalid new shape of the convolution')

    def backwardpass(self, x, delta, batch_length):
        if self.input_layer:
            delta_kernel = cc.backwardConvolutionW(x, delta, self.stride, self.kernel.shape)
            delta_x = None
            delta_kernel = np.array(delta_kernel)
            delta_kernel = batch_length * delta_kernel
        else:
            delta_x = cc.backwardConvolutionX(x, delta, self.kernel, self.stride)
            delta_kernel = cc.backwardConvolutionW(x, delta, self.stride, self.kernel.shape)
            delta_x = np.array(delta_x)
            delta_kernel = np.array(delta_kernel)
            delta_kernel = batch_length * delta_kernel

        return delta_kernel, delta_x


class output_layer():

    def __init__(self, n_conn, n_neur, func):
        self.b = np.random.normal(0, .05, (1, n_neur))
        self.W = np.random.normal(0, .05, (n_conn, n_neur))
        self.func = func

    def activation(self, x):
        self.z = (x @ self.W) + self.b
        self.a = self.func[0](self.z)

    def backwardpass(self, x, output, batch_length):
        d = self.a - output
        delta = batch_length * (x.T @ d)
        return d, delta


class neural_layer():

    def __init__(self, n_conn, n_neur, func):
        self.b = np.random.normal(0, .05, (1, n_neur))
        self.W = np.random.normal(0, .05, (n_conn, n_neur))
        self.func = func

    def activation(self, x):
        self.z = (x @ self.W) + self.b
        self.a = self.func[0](self.z)
        self.derivative = self.func[1](self.z)

    def backwardpass(self, x, weights, dX, batch_length):
        d = (dX @ weights.T) * self.derivative
        delta = batch_length * (x.T @ d)
        return d, delta
