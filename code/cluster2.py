import numpy as np

def _naive_LCSS(F1, F2, epsilon, delta):
    """
    takes
    F1: np.array([N, 2])
    F2: np.array([N, 2])
    epsilon: float
    delta: int
    
    """

    T1 = len(F1)
    T2 = len(F2)

    norm = lambda x: np.sqrt(np.dot(x,x))
    if T1 == 0 and T2 == 0:
        return 0
    elif norm(F1[-1] - F2[-1]) < epsilion and np.abs(T1-T2) < delta:
        return 1 + naive_LCSS(F1[:-1], F2[:-1])
    else:
        return max(naive_LCSS(F1[:-1], F2), naive_LCSS(F1, F1[:-1]))

def LCSS(F1, F2, epsilon, delta):
    return 1 - float(naive_LCSS(F1, F2, epsilon, delta))/min(len(F1), len(F2))

def cluster(curves):
    """

    """
