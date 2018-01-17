import numpy as np
import scipy as sp

def UpdateTheta(K, C, Y, rho):
    par1 = 1 / rho * (K + Y - rho * C)
    LamArray, eigVec = np.linalg.eigh(par1)
    LamArray = (-rho * LamArray + np.sqrt(np.square(rho * LamArray) + 4 * rho)) / rho / 2
    newTheta = np.diag(LamArray)
    newTheta = np.matmul(np.matmul(eigVec, newTheta), np.transpose(eigVec))
    return(newTheta.copy())

def UpdateC(Theta, Y, A, rho):
    N = np.shape(A)[1]
    newC = Y / rho + Theta
    offDiagA = A.copy()
    offDiagA[np.diag_indices(N)] = 0
    B = np.ones((N, N)) - offDiagA - np.identity(N)
    newC[np.logical_and(A == 1, newC > 0)] = 0
    newC[B == 1] = 0
    newC[np.logical_and(np.identity(N) == 1, newC < 0)] = 0
    return(newC.copy())

def UpdateY(Theta, C, rho, lastY):
    newY = lastY + rho * (Theta - C)
    return(newY.copy())