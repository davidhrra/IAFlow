import IAFlow as iaf
import numpy as np
from utils.mnist_reader import load_mnist
import matplotlib.pyplot as plt
from time import time
from scipy import signal
import scipy as sc
import math
import sys

def saveParams(params, name):
    np.save(name, params)

def load_params(name):
    loaded = np.load(name)
    return loaded

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def toRaw(data):
    yProcessed = []
    for i in data:
        temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        temp[i] = 1
        yProcessed.append(temp)
    yProcessed = np.array(yProcessed, dtype='float64')
    return yProcessed

def flatter(x):
    flattenedExample = []
    for example in x:
        flattened = []
        for filter in example:
            flattened.append(np.reshape(filter, (filter.shape[0]*filter.shape[1])))
        flattenedExample.append(np.reshape(flattened, len(flattened)*len(flattened[1])))
    flattened = np.array(flattenedExample)
    return flattened

def normalRearrange(x):
    rearranged = []
    for example in x:
        rearranged.append(np.reshape(example, (int(example.shape[0]**(1/2)), int(example.shape[0]**(1/2)))))
    rearranged = np.array(rearranged)
    return rearranged


def rearrange(x, filters):
    rearranged = []
    for example in x:
        rearranged.append(np.array(normalRearrange(np.reshape(example, (filters, int(example.shape[0]/filters))))))
    rearranged = np.array(rearranged)
    return rearranged

def checkGradients(layer, x, y_train):
    epsilon = 1e-4

    Delta_checked = np.zeros(shape=layer.W.shape)

    Delta_checked = Delta_checked.flatten()
    Delta_real = np.array(layer.W.flatten())

    for i in range(Delta_checked.shape[0]):
        print(i)
        Delta_copy = Delta_real
        Delta_copy[i] += epsilon
        layer.W = np.reshape(Delta_copy, layer.W.shape)
        a5 = feedforward(x)
        Jpos = iaf.classificationCost(a5, y_train)

        Delta_copy = Delta_real
        Delta_copy[i] -= epsilon
        layer.W = np.reshape(Delta_copy, layer.W.shape)
        a5 = feedforward(x)
        Jneg = iaf.classificationCost(a5, y_train)

        Delta_checked[i] = (Jpos - Jneg) / (2 * epsilon)

    layer.W = Delta_real
    Delta_checked = np.reshape(Delta_checked, layer.W.shape)

    return Delta_checked


def Backpropagation(x, y):
    batch_size = 1 / y.shape[0]

    d5, Delta5 = layer_5.backwardpass(layer_4.a, y, batch_size)
    d4, Delta4 = layer_4.backwardpass(np.reshape(layer_3.pooledArray,(layer_3.pooledArray.shape[0], layer_3.pooledArray.shape[1]* \
                                             layer_3.pooledArray.shape[2]*layer_3.pooledArray.shape[3])), \
                                      layer_5.W, d5, batch_size)
    d4 = (d4 @ layer_4.W.T)
    d4 = np.reshape(d4, (layer_3.pooledArray.shape))
    d3 = layer_3.backwardpass(layer_2.activation, d4)
    d3 = d3 * layer_2.func[1](layer_2.convolved)
    Delta2, d2 = layer_2.backwardpass(layer_1.activation, d3, batch_size)
    d2 = d2 * layer_1.func[1](layer_1.convolved)
    Delta1, _ = layer_1.backwardpass(x, d2, batch_size)

    return Delta1, Delta2, Delta4, Delta5


def sgd(epoch, rate, x, batch_size, decay, y, momentum):

    iterations = int(x.shape[0] / batch_size)

    for i in range(epoch):
        x_sorted, y_sorted = unison_shuffled_copies(x, y)
        vDelta1 = 0
        vDelta2 = 0
        vDelta4 = 0
        vDelta5 = 0
        for batch in range(iterations):

            x_train = x_sorted[batch*batch_size:(batch+1)*batch_size,:,:,:]
            y_train = y_sorted[batch*batch_size:(batch+1)*batch_size,:]

            a5 = feedforward(x_train)

            Delta1, Delta2, Delta4, Delta5 = Backpropagation(x_train, y_train)

            threshold = 2

            preV_Delta1 = vDelta1
            preV_Delta2 = vDelta2
            preV_Delta4 = vDelta4
            preV_Delta5 = vDelta5

            vDelta1 = (momentum * vDelta1) - (rate * Delta1)
            vDelta2 = (momentum * vDelta2) - (rate * Delta2)
            vDelta4 = (momentum * vDelta4) - (rate * Delta4)
            vDelta5 = (momentum * vDelta5) - (rate * Delta5)

            update_Delta1 = - momentum * preV_Delta1 + (1 + momentum) * vDelta1
            update_Delta2 = - momentum * preV_Delta2 + (1 + momentum) * vDelta2
            update_Delta4 = - momentum * preV_Delta4 + (1 + momentum) * vDelta4
            update_Delta5 = - momentum * preV_Delta5 + (1 + momentum) * vDelta5

            '''if abs(update_Delta1.max()) or abs(update_Delta1.min()) >= threshold:
                update_Delta1 = iaf.gradientClipping(update_Delta1, threshold)
            if abs(update_Delta2.max()) or abs(update_Delta2.min()) >= threshold:
                update_Delta2 = iaf.gradientClipping(update_Delta2, threshold)
            if abs(update_Delta4.max()) or abs(update_Delta4.min()) >= threshold:
                update_Delta4 = iaf.gradientClipping(update_Delta4, threshold)
            if abs(update_Delta5.max()) or abs(update_Delta5.min()) >= threshold:
                update_Delta5 = iaf.gradientClipping(update_Delta5, threshold)'''

            '''update_Delta5 = Delta5
            update_Delta4 = Delta4
            update_Delta2 = Delta2
            update_Delta1 = Delta1'''

            #Delta5_checked = checkGradients(layer_5, x, y_train)

            layer_5.W += update_Delta5
            layer_5.b += np.mean(update_Delta5, axis=0, keepdims=True)
            layer_4.W += update_Delta4
            layer_4.b += np.mean(update_Delta4, axis=0, keepdims=True)
            layer_2.kernel += update_Delta2
            layer_2_bias = np.reshape(np.mean(update_Delta2, axis=(0, 1, 2), keepdims=True),layer_2.kernel.shape[3])
            layer_2.bias += layer_2_bias
            layer_1.kernel += update_Delta1
            layer_1_bias = np.reshape(np.mean(update_Delta1, axis=(0, 1, 2), keepdims=True), layer_1.kernel.shape[3])
            layer_1.bias += layer_1_bias

            J = iaf.classificationCost(a5, y_train)

            value = percentage(y_train, a5)


            print('Epoch: %i \t Batch: %i-%i \t Cost Function: %f \t Accuracy: %.2f%%'\
                  % (i+1, batch*batch_size, (batch+1)*batch_size, J, value*100))

            if math.isnan(J):
                sys.exit()
        rate -= decay

def feedforward(x):

    layer_1.forwardpass(x)
    a1 = layer_1.activation
    layer_2.forwardpass(a1)
    a2 = layer_2.activation
    layer_3.forwardpass(a2)
    a3 = np.reshape(layer_3.pooledArray,(layer_3.pooledArray.shape[0], layer_3.pooledArray.shape[1]* \
                                             layer_3.pooledArray.shape[2]*layer_3.pooledArray.shape[3]))
    layer_4.activation(a3)
    a4 = layer_4.a
    layer_5.activation(a4)
    a5 = layer_5.a

    return a5

def percentage(y_real, y_predict):
    count = 0
    for i in range(y_predict.shape[0]):
        if y_predict[i].argmax() == y_real[i].argmax():
            count += 1
    value = count/len(y_predict)
    return value


if __name__ == '__main__':

    x_train, y_train = load_mnist('data/fashion', kind='train')
    x_test, y_test = load_mnist('data/fashion', kind='t10k')

    layer_1 = iaf.convolutional_layer(h=3, w=3, func=iaf.leakyReLU,
                                          filters=8, channels=1, stride=1, input_layer=True)
    layer_2 = iaf.convolutional_layer(h=3, w=3, func=iaf.leakyReLU,
                                          filters=16, channels=8, stride=1)
    layer_3 = iaf.maxPooling(2, 2)
    layer_4 = iaf.neural_layer(2304, 512, iaf.leakyReLU)
    layer_5 = iaf.output_layer(512, 10, iaf.softmax)

    params = load_params('NistParams5.npy')

    layer_1.kernel = np.array(params[0], dtype='float64')
    layer_1.bias = np.array(params[1], dtype='float64')
    layer_2.kernel = np.array(params[2], dtype='float64')
    layer_2.bias = np.array(params[3], dtype='float64')
    layer_4.W = np.array(params[4], dtype='float64')
    layer_4.b = np.array(params[5], dtype='float64')
    layer_5.W = np.array(params[6], dtype='float64')
    layer_5.b = np.array(params[7], dtype='float64')

    y = toRaw(y_test)
    x = np.reshape(x_test, (10000, 28, 28, 1))
    x = x.astype('double', copy=False)
    x /= 255

    a5 = feedforward(x)

    J = iaf.classificationCost(a5, y)

    print('The cost function for the test set was: %f' %(J))

    value = percentage(y, a5)
    print('%.2f %% of examples were classified correctly' % (value * 100))

    '''for i in range(10):
        pos = np.random.randint(0, 10000)
        show_images([layer_1.activation[pos, :, :, 0], layer_1.activation[pos, :, :, 1],
                     layer_1.activation[pos, :, :, 2], layer_1.activation[pos, :, :, 3],
                     layer_1.activation[pos, :, :, 4], layer_1.activation[pos, :, :, 5],
                     layer_1.activation[pos, :, :, 6], layer_1.activation[pos, :, :, 7],
                     layer_2.activation[pos, :, :, 0], layer_2.activation[pos, :, :, 1],
                     layer_2.activation[pos, :, :, 2], layer_2.activation[pos, :, :, 3],
                     layer_2.activation[pos, :, :, 4], layer_2.activation[pos, :, :, 5],
                     layer_2.activation[pos, :, :, 6], layer_2.activation[pos, :, :, 7],
                     layer_2.activation[pos, :, :, 8], layer_2.activation[pos, :, :, 9],
                     layer_2.activation[pos, :, :, 10], layer_2.activation[pos, :, :, 11],
                     layer_2.activation[pos, :, :, 12], layer_2.activation[pos, :, :, 13],
                     layer_2.activation[pos, :, :, 14], layer_2.activation[pos, :, :, 15]], 3)


    start_time = time()
    epochs = 5
    rate = .04

    y = toRaw(y_train)
    x = np.reshape(x_train, (60000, 28, 28, 1))
    x = x.astype('float', copy=False)
    x /= 255

    #Backpropagation & Gradient Descent
    sgd(epochs, rate, x, 500, 0.0, y, 0.9)

    final_time = time()
    total_time = final_time - start_time
    print('The total train time was %fs' % (total_time))

    y_test = toRaw(y_test)
    x_test = np.reshape(x_test, (10000, 28, 28, 1))
    x_test = x_test.astype('float', copy=False)
    x_test /= 255

    a5 = feedforward(x_test)
    value = percentage(y_test, a5)
    print('%.2f %% of examples were classified correctly'%(value*100))
    J = iaf.classificationCost(a5, y_test)
    print('cost function at the test set: %f'%(J))'''

    for i in range(10):
        pos = np.random.randint(0, 5000)
        show_images([layer_1.activation[pos, :, :, 0], layer_1.activation[pos, :, :, 1],
                     layer_1.activation[pos, :, :, 2], layer_1.activation[pos, :, :, 3],
                     layer_1.activation[pos, :, :, 4], layer_1.activation[pos, :, :, 5],
                     layer_1.activation[pos, :, :, 6], layer_1.activation[pos, :, :, 7]], 8)


    #saveParams([layer_1.kernel, layer_1.bias, layer_2.kernel,  layer_2.bias, layer_4.W, layer_4.b, layer_5.W, layer_5.b], 'NistParams5')