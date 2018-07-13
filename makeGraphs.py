import os,sys;sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


for i in range(3):
    file = open("epGreedyAnnealingScaledResults" + str(i) + ".txt","r")
    file.readline()
    data = file.read()
    print(data)
    data = data.split("\n")
    for j in range(len(data)):
        #print(data[j])
        data[j] = data[j].split()
        
    fig = plt.figure()
    print("test")
    
    #fig.set_title("Z Coords")
    #fig.set_xlabel("t")
    #fig.set_ylabel("Z Approximation")
    
    numbers = np.zeros(len(data))
    zs = np.zeros(len(data))
    for j in range(len(data)):
        if (data[j][0] != ''):
            zs[j] = float(data[j][6])
            numbers[j] = j
    #print(data)
    
    mpl.pyplot.plot(numbers, zs)
    mpl.pyplot.xlabel("t")
    mpl.pyplot.ylabel("Z Values")
    fig.show()
    fig.savefig("zAnnealingScaled" + str(i) + ".png")
    
    file.close()
    


print("done")