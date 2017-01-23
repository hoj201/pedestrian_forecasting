#!/usr/bin/env python
import numpy as np

def cdf_of_normal(mu, sigma, x_min, x_max):
    """ Computes the integral of a normal distribution over a bounding box.

    args:
        mu: numpy array of size 2
        sigma: float, standard deviation
        x_min: float
        x_max: float

    returns:
        out: float
    """
    from scipy.special import erf
    out = erf( (x_max-mu)/(np.sqrt(2)*sigma) )
    out -= erf( (x_min-mu)/(np.sqrt(2)*sigma) )
    out *= 0.5
    return out

#TODO: Consider vectorizing this
def cdf_of_2d_normal(mu, sigma, bbox):
    """ Computes the integral of a normal distribution over a bounding box.

    args:
        mu: numpy array of size 2, mean of normal distribution
        sigma: float, standard deviation
        bbox: 4-tuple (x_min, x_max, y_min, y_max)

    returns:
        out: float
    """
    x_min, x_max, y_min, y_max = bbox
    out_x = cdf_of_normal(mu[0], sigma, x_min, x_max)
    out_y = cdf_of_normal(mu[1], sigma, y_min, y_max)
    out = out_x * out_y
    return out

if __name__ == "__main__":
    print "Testing routine cdf_of_normal"
    sigma = 1.1
    mu = 0.0
    x_min = -0.2
    x_max = 1
    result = cdf_of_normal(mu, sigma, x_min, x_max)
    from scipy.integrate import quad
    def integrand(x):
        return np.exp( -(x-mu)**2 / (2*sigma**2) ) / (np.sqrt(2*np.pi*sigma**2))
    expected, error = quad( integrand, x_min, x_max )
    print "result   = {}".format(result)
    print "expected = {} +/- {}".format(expected, error)

    print "Testing integration_of_Gaussian_over_bbox"
    sigma = 1.1
    mu = np.random.randn(2)
    bbox = (-1.0, 5.1, -1, 0)
    result = cdf_of_2d_normal(mu, sigma, bbox)
    from scipy.integrate import dblquad
    def integrand(y,x): #NOTE: dblequad uses this stupid convention where they swap the order of x and y in the integrand
        out = np.exp( -((x-mu[0])**2+(y-mu[1])**2)/(2*sigma**2))
        out /= 2*np.pi*sigma**2
        return out
    expected, error = dblquad(
            integrand, bbox[0], bbox[1], lambda x: bbox[2], lambda x: bbox[3]
            )
    print "result   = {}".format(result)
    print "expected = {} +/- {}".format(expected, error)
