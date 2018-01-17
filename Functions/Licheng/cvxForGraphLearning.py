import numpy as np
from cvxopt import *
from cvxpy import *
import mosek

def cvxSolver(A, K):
    #Use official CVX to get the solution
    N = np.shape(A)[1]
    offDiagA = A.copy()
    offDiagA[np.diag_indices(N)] = 0
    const0 = np.ones((N, N)) - offDiagA - np.identity(N)

    Theta = semidefinite(N)
    objective = Minimize(-log_det(Theta) + trace(K * Theta))
    constraints = [
                     mul_elemwise(offDiagA, Theta) <= 0,
                     mul_elemwise(const0, Theta) == 0
                  ]
    prob = Problem(objective, constraints)
    prob.solve(solver = SCS)
    print(prob.value)
    finalTheta = np.multiply(Theta.value, (np.ones((N, N)) - const0))
    finalTheta[np.logical_and(offDiagA == 1, finalTheta > 0)] = 0
    return(finalTheta)