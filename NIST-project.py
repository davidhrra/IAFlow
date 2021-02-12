import IAFlow as iaf
from scipy import ndimage
from matplotlib.pyplot import imshow, show
import numpy as np
from utils.mnist_reader import load_mnist

def show_data(data):
    data = np.reshape(data, (28, 28))
    imshow(X=data, cmap='gray')
    show()

def toRaw(data):
    yProcessed = []
    for i in data:
        temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        temp[i - 1] = 1
        yProcessed.append(temp)
    yProcessed = np.array(yProcessed, dtype='float64')
    return yProcessed

def fromRaw(data):
    yProcessed = []
    for i in data:
        for j in range(len(i)):
            if i[j] == 1:
                yProcessed.append(j+1)

    yProcessed = np.array(yProcessed)
    return yProcessed

def max(data):
    yNorm = np.zeros(data.shape)
    for i in range(len(data)):
        yNorm[i][np.argmax(data[i])] = 1
    return yNorm

def percentage(y_real, y_predict):
    count = 0
    for i in range(y_predict.shape[0]):
        if y_predict[i].argmax() == y_real[i].argmax():
            count += 1
    value = count/y_real.shape[0]
    return value

if __name__ == '__main__':

    X_train, y_train = load_mnist('data/fashion', kind='train')
    X_test, y_test = load_mnist('data/fashion', kind='t10k' )

    X_train = np.array(X_train, dtype='float64')
    yRaw = toRaw(y_train)

    X_train = X_train
    yRaw = yRaw

    nn = iaf.snn([784, 676, 24, 10], iaf.sigm, iaf.sigm, 'classificationwReg', 0)
    nn.insertData(X_train, yRaw)
    nn.feedforward()
    cost = iaf.classificationCostwReg(nn.weights, nn.z, nn.Y, 0)
    print('The cost function is %f'% (cost))
    nn.backpropagation()
    print(nn.z[3].shape)
    nn.gradientDescent(0.1, 40)
    cost = iaf.classificationCostwReg(nn.weights, nn.z, nn.Y, 0)
    print('The cost function is %f'% (cost))
    yFinal = nn.z[3]
    corrects = np.round(percentage(y_train, yFinal) * 100, 2)
    print('The model classified good %.2f of examples'% (corrects))




