import numpy as np

def objValue(Theta, K):
    objValue = -np.log(np.linalg.det(Theta)) + np.trace(np.matmul(K, Theta))
    return(objValue)