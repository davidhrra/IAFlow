import numpy as np
from matplotlib.pyplot import plot, show
from sys import exit

def load_data():
    data = np.loadtxt("Datos_2.txt", dtype='float32')
    return data

def sigmoid_function(z):
    a = 1/(1+np.exp(-z))
    return a

def cost_function(x, y, thetas):
    a2, a3, a4 = feeding(x, thetas)
    a4 = [(x+1)/2 for x in a4]
    J = (1/len(y))*np.sum((np.multiply(-y, np.log(a4))-(np.multiply(1-y, np.log(np.subtract(1, a4))))))
    return J

'''De la retroalimentación (backpropagation) obtenemos la derivada parcial de J(THETA) o el "grado de error" 
de cada uno de los nodos de Theta'''
def backpropagation(x, a, y, thetas):
    a4 = a[2]
    d4 = a4-y

    temp = np.subtract(1,np.power(a3, 2))
    d3 = np.dot(d4, thetas[2])
    d3 = np.multiply(d3, temp)
    d3 = np.delete(d3, [0], 1)

    temp = np.subtract(1, np.power(a[0], 2))
    d2 = np.dot(d3, thetas[1])
    d2 = np.multiply(d2, temp)
    d2 = np.delete(d2, [0], 1)

    Delta3 = np.dot(d4.transpose(), a[1])
    Delta2 = np.dot(d3.transpose(), a[0])
    Delta1 = np.dot(d2.transpose(), x)

    Delta3 = np.multiply(1 / len(y), Delta3)
    Delta2 = np.multiply(1 / len(y), Delta2)
    Delta1 = np.multiply(1 / len(y), Delta1)

    return Delta1, Delta2, Delta3

def gradient_descent(x, y, theta, learningRate, iteration):
    print("Los valores de Theta son:")
    print(theta)
    J_History = []
    iter = []
    for i in range(iteration):
        minTheta1 = np.resize(theta[0:12], (4, 3))
        minTheta2 = np.resize(theta[12:20], (2, 4))
        minTheta3 = np.resize(theta[20:23], (1, 3))
        thetaSend = [minTheta1, minTheta2, minTheta3]

        a2, a3, a4 = feeding(x, thetaSend)
        Delta1, Delta2, Delta3 = backpropagation(x, [a2, a3, a4], y, thetaSend)

        DeltaUnrolled = np.resize(Delta1, (len(Delta1)*len(Delta1[0]), 1))
        DeltaUnrolled = np.append(DeltaUnrolled, np.resize(Delta2, (len(Delta2) * len(Delta2[0]), 1)))
        DeltaUnrolled = np.append(DeltaUnrolled, np.resize(Delta3, (len(Delta3) * len(Delta3[0]), 1)))
        DeltaUnrolled = np.resize(DeltaUnrolled, (len(DeltaUnrolled), 1))

        theta -= np.multiply(learningRate, DeltaUnrolled)
        thetaCost1 = np.resize(theta[0:12], (4, 3))
        thetaCost2 = np.resize(theta[12:20], (2, 4))
        thetaCost3 = np.resize(theta[20:23], (1, 3))

        J = cost_function(x, y, [thetaCost1, thetaCost2, thetaCost3])
        J_History.append(J)
        iter.append(i+1)
        print("The cost function at the %i iteration is %f" % (i, J))
        print(theta)
    plot(iter, J_History)
    show()
    return theta

def feeding(x, thetas):
    g2 = np.dot(x, thetas[0])
    a2 = np.insert(np.tanh(g2), [0], np.ones((len(g2),1)),axis=1)
    g3 = np.dot(a2, np.transpose(thetas[1]))
    a3 = np.insert(np.tanh(g3), [0], np.ones((len(g3),1)),axis=1)
    g4 = np.dot(a3, np.transpose(thetas[2]))
    a4 = np.tanh(g4)
    return a2, a3, a4

def predict(x, theta):
    a2, a3, a4=feeding(x, theta)
    prediction = []
    for i in a4:
        if(i >= 0.5):
            prediction.append(True)
        else:
            prediction.append(False)
    return prediction

def verification(xresult, y):
    count = 0
    for i in range(len(xresult)):
        if xresult[i] == y[i]:
            count +=1
    percentage = (count/len(xresult))*100
    return percentage

def saveTheta(theta):
    np.save("thetaSavesTanh", theta)

    thetaUnrolled = np.resize(theta[0], (len(theta[0]) * len(theta[0][0]), 1))
    thetaUnrolled = np.append(thetaUnrolled, np.resize(theta[1], (len(theta[1]) * len(theta[1][0]), 1)))
    thetaUnrolled = np.append(thetaUnrolled, np.resize(theta[2], (len(theta[2]) * len(theta[2][0]), 1)))
    thetaUnrolled = np.resize(thetaUnrolled, (len(thetaUnrolled), 1))

    np.savetxt("thetaSavesTanh.txt", thetaUnrolled)
    pass

def loadTheta():
    theta = np.load("thetaSaves.npy")
    return theta

if __name__ == "__main__":
    theta1 = np.multiply(0.5, np.random.randn(4, 3))
    theta2 = np.multiply(0.1, np.random.randn(2, 4))
    theta3 = np.multiply(0.3, np.random.randn(1, 3))
    thetaMatrix = [theta1, theta2, theta3]
    grad = True
    #thetaMatrix = loadTheta()
    resources = load_data()

    x = np.insert(np.resize(resources[:,0:len(resources[0])-1], (len(resources), len(resources[0])-1)), [0], np.ones((len(resources),1)), axis=1)
    y = np.resize(resources[0:len(resources), 3], (len(resources), 1))

    a2, a3, a4 = feeding(x, thetaMatrix)
    cost = cost_function(x, y, thetaMatrix)
    print("The cost of the cost function is %f" % (cost))
    Delta1, Delta2, Delta3 = backpropagation(x, [a2, a3, a4],y, thetaMatrix)

    if grad:
        thetaInitialUnrolled = np.resize(thetaMatrix[0], (len(thetaMatrix[0]) * len(thetaMatrix[0][0]), 1))
        thetaInitialUnrolled = np.append(thetaInitialUnrolled,
                                         np.resize(thetaMatrix[1], (len(thetaMatrix[1]) * len(thetaMatrix[1][0]), 1)))
        thetaInitialUnrolled = np.append(thetaInitialUnrolled,
                                         np.resize(thetaMatrix[2], (len(thetaMatrix[2]) * len(thetaMatrix[2][0]), 1)))
        thetaInitialUnrolled = np.resize(thetaInitialUnrolled, (len(thetaInitialUnrolled), 1))

        minTheta = gradient_descent(x, y, thetaInitialUnrolled, 0.08, 45000)

        minTheta1 = np.resize(minTheta[0:12], (4, 3))
        minTheta2 = np.resize(minTheta[12:20], (2, 4))
        minTheta3 = np.resize(minTheta[20:23], (1, 3))

        minTheta = [minTheta1, minTheta2, minTheta3]
    else:
        minTheta = thetaMatrix

    result = predict(x, minTheta)

    xCorrect = verification(result, y)
    print("The %f %% of training examples were classified correctly" % (xCorrect))

    valData= np.loadtxt("validationData.txt", dtype='float32')

    xVal = np.insert(np.resize(valData[:, 0:len(valData[0]) - 1], (len(valData), len(valData[0]) - 1)), [0],
                  np.ones((len(valData), 1)), axis=1)
    yVal = np.resize(valData[0:len(valData), 3], (len(valData), 1))

    resultVal = predict(xVal, minTheta)
    correctVal = verification(resultVal, yVal)

    print("The %f %% of new examples were classified correctly" % (correctVal))
    print("The average efficiency for the Neural Network is %f %%" % ((correctVal+xCorrect)/2))
    saveTheta(minTheta)
