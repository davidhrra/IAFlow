import IAFlow as flw
from scipy import ndimage
from matplotlib.pyplot import imshow, show
import random
import numpy as np

def load_data():
    data = np.loadtxt("Data.txt", dtype='float32')
    return data

def num(y):
    resultRaw = []
    for i in range(len(y)):
        resultRaw.append(np.where(y[i] == y[i].max())[0]+1)
    return resultRaw

def comp(y, result):
    count = 0
    for i in range(len(result)):
        if np.argmax(y[i]) == np.argmax(result[i]):
            count += 1
    percentage = count / len(result)
    return percentage

def plots(vector):
    matrix = np.reshape(vector, (20, 20))
    matrix = ndimage.rotate(matrix, 90)
    imshow(matrix, cmap='gray', origin='lower')
    show()

def toRaw(data):
    yProcessed = []
    for i in data:
        temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        temp[int(i) - 1] = 1
        yProcessed.append(temp)
    yProcessed = np.array(yProcessed, dtype='float64')
    return yProcessed

if __name__ == "__main__":
    pNN = flw.snn([400, 25, 10], flw.relu, flw.softmax, 'regression', 0)

    preResources = load_data()
    randomize = [x for x in range(5000)]
    random.shuffle(randomize)
    resources = []
    for i in range(len(randomize)):
        resources.append(preResources[randomize[i]])
    resources = np.array(resources)
    xTrain = np.resize(resources[0:4500, 0:len(resources[0])-1], (4500,400))
    rawY = np.resize(resources[0:4500, 400], (4500, 1))
    yTrain = toRaw(rawY)


    pNN.insertData(xTrain, yTrain)
    pNN.feedforward()
    pNN.backpropagation()
    pNN.gradientDescent(0.1, 200)

    train_result = pNN.z[2]

    efficiency = comp(train_result, yTrain)
    efficiency *= 100
    print("The efficiency in the train set was of %.2f %%" % (efficiency))

    xTest = np.resize(resources[4500:, 0:len(resources[0]) - 1], (500, 400))
    rawYTest = np.resize(resources[4500:, 400], (500, 1))
    yTest = toRaw(rawYTest)

    pNN.feedforward(xTest)
    test_result = pNN.z[2]

    efficiency = comp(test_result, yTest)
    efficiency *= 100
    print("The efficiency in the validation set was of %.2f %%" % (efficiency))
    rep = 1
    #while rep != 0:
        #pos = random.randint(0,500)
        #print("El número predicho por la red neuronal es %i" % (rawXval[pos]))
        #plots(rawResult[pos])
        #rep = input("Desea predecir otro número? si = 1 | no = 0")