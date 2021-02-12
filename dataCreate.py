import numpy as np

trues = 0
meanArray = []
for i in range(2000):
    mean = np.absolute(np.round(np.random.normal(3.5,0.3, (1,3)), decimals=1))
    final = np.sum(mean)/3
    if final >= 3.5:
        finalAppend = np.append(mean, True)
        trues+=1
    else:
        finalAppend = np.append(mean, False)
    meanArray.append(finalAppend)

np.savetxt("validationData.txt", meanArray, fmt='%4G')
print(trues)
print(np.max(meanArray))
print(np.min(meanArray))