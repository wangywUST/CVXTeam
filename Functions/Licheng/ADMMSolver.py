import numpy as np
from UpdateTheta import *
from objValue import *
import matplotlib.pyplot as plt

def ADMMSolver(A, K, rho):
    N = np.shape(A)[1]
    lastY = np.identity(N)
    lastC = np.identity(N)
    lastTheta = np.identity(N)
    count = 0
    objArray = []
    while (count < 100):
        count = count + 1
        newTheta = UpdateTheta(K, lastC, lastY, rho)
        newC = UpdateC(newTheta, lastY, A, rho)
        newY = UpdateY(newTheta, newC, rho, lastY)

        print(np.linalg.norm(lastTheta - newTheta, 'fro'))
        print(count)
        lastTheta = newTheta.copy()
        lastC = newC.copy()
        lastY = newY.copy()
        objArray = np.append(objArray, objValue(lastTheta, K))
#        print(objValue(lastTheta, K))

    return(lastTheta)

