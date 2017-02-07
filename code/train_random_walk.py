#!/usr/bin/env python
import numpy as np

def NLL(x,sigma):
    """ Computes the negative log-liklihood of a path,
    assuming it came from a random walk of variance sigma.

    args:
    x: np.array, x.shape == (d,N)
    sigma: float, sigma > 0
    
    returns:
    out: float, out >= 0

    NOTE: We neglect a constant term N*d*log(2*pi)/2
    """
    
    d,N = x.shape
    dx = x[1:] - x[:N-1]
    out = np.dot(dx,dx).sum() / (2*sigma**2)
    out += N * d * np.log(sigma)
    return out


def jac_NLL(x,sigma):
    """ Computes the derivative of NLL(x,sigma) wrt sigma

    args:
    x: np.array, x.shape == (d,N)
    sigma: float, sigma>0

    returns:
    out: float
    """
    
    d,N = x.shape
    dx = x[:,1:] - x[:,:N-1]
    out = -2*np.dot(dx,dx).sum() / sigma**3
    out += N*d / sigma
    return out

def compute_var(x):
    """ Computes the variance of a single path

    args:
    x: np.array, x.shape == (d,N)

    returns:
    variance: float > 0
    """
    d,N = x.shape
    dx = x[:,1:] - x[:,:N-1]
    ss = (dx[0]**2 + dx[1]**2).sum()
    return ss / (N*d)

def learn_sigma_RW(x_ls):
    """ Learns standard deviation of a random walk, from a list of paths

    args:
        x_ls: list of np.arrays. x_ls[i].shape = (d,N_i)

    returns:
        sigma: positive float
    """
    var_ls = [ compute_var(x) for x in x_ls]
    N_ls = [x.shape[1] for x in x_ls]
    N_tot = sum(N_ls)
    variance = sum([ N*var/N_tot for (N,var) in zip(N_ls, var_ls)])
    return np.sqrt(variance)

if __name__ == "__main__":
    sigma_true = 0.01
    N_ls = [np.random.randint(100,600) for _ in range(10)]
    dx_ls = [np.random.randn(2,N)*sigma_true for N in N_ls]
    x_ls = [np.stack([dx[0].cumsum(), dx[1].cumsum()]) for dx in dx_ls]
    print "Computed = {}".format(learn_sigma_RW(x_ls))
    print "Expected = {}".format( sigma_true )
