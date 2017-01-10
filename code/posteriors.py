import numpy as np
#from energy import Vk
from numpy.polynomial import legendre
from integrate import trap_quad
from scipy.stats import multivariate_normal
import pickle
from scene import Scene


with open("test_scene.pkl", "rb") as f:
    scene = pickle.load(f)
Vk = scene.alpha_arr
scene_scale = np.array([scene.width, scene.height])
#temporary
dist_width = np.ones([2]) * scene.bbox_width
vel_width = np.ones([2]) * scene.bbox_velocity_width
s_max = scene.s_max
sigma_x = scene.sigma_x
sigma_v = scene.sigma_v
sigma_L = scene.sigma_L

def legval_scale(x, y, coeffs):
    """
    Returns the values of a legend repolynomial
    at N points, scaled so the whole plane is [0,1]**2

    Takes x: np.array(N_Points): x values to evaluate
    Takes y: np.array(N_Points): y values to evaluate
    Takes coeffs: np.array(K_max_degree): legendre polynomial coefficients

    Returns np.array(N_Points)

    """
    out = legendre.legval2d(x/scene_scale[0], y/scene_scale[1], coeffs)
    out -= out.min()
    return out

def x_hat_given_x(x_hat, x):
    """
    Returns the probability of a measurement of x_hat given x.

    Takes x_hat: np.array(2): initial point
    Takes x: np.array(N_points, 2): set of points at which to evaluate probability

    Returns np.array(N_Points): probability that agent is at x given x0

    """
    exponent = - (x_hat[0] - x[0])**2 - (x_hat[1] - x[1])**2
    exponent /= 2*sigma_x**2
    return np.exp(exponent) / (2*np.pi*sigma_x**2)


def v_hat_given_v(v_hat, v):
    """
    Returns the probability of a measurement of v_hat given v.

    Takes v_hat: np.array(2): initial point
    Takes v: np.array(N_points, 2): set of points at which to evaluate probability

    Returns np.array(N_Points): probability that agent is at x given x0

    """
    exponent = - (v_hat[0] - v[0])**2 - (v_hat[1] - v[1])**2
    exponent /= 2*sigma_v**2
    return np.exp(exponent) / (2*np.pi*sigma_v**2)


def prob_s_uniform(s):
    """
    Returns P(s) assuming a uniform distribution of +-s_max
    s_max: float
    s: np.array(n_points): s values to be sampled
    """
    ident = s <= s_max
    return 1.0/(2*s_max) * ident


_normalization_constants = []
_bounds = (-0.5*scene_scale[0], 0.5*scene_scale[0], -0.5*scene_scale[1], 0.5*scene_scale[1])

for coeffs in Vk:
    fn = lambda x, y: np.exp( -1*legval_scale(x, y, coeffs))
    _normalization_constants.append(trap_quad(fn, _bounds, res=(200,200)))

def x_given_k(x,k):
    """
    Returns probability that an agent will be at x given k

    Takes k: int: cluster
    Takes x: np.array(2, N_points)
    Returns np.array(N_points)
    According to the equation (1/Z_K)exp(-1*V_k)
    """
    out = (1.0/_normalization_constants[k])
    out *= np.exp( -1*legval_scale(x[0], x[1], Vk[k]))
    out *= (x[0] <= scene_scale[0]/2.0)*(x[0] >= -scene_scale[0]/2.0)
    out *= (x[1] <= scene_scale[1]/2.0)*(x[1] >= -scene_scale[1]/2.0)
    #out = np.ones(x.shape[1])
    return out

def x_given_lin(x):
    """
    takes x: np.array(2, N_points): points to be evaluated

    Returns np.array(N_points)
    """
    if len(x.shape) == 1:
        const_arr = 1.0
    else:
        const_arr = np.ones(x.shape[1])
    out = const_arr/(scene_scale[0] * scene_scale[1])
    out *= (x[0] <= scene_scale[0]/2.0)*(x[0] >= -scene_scale[0]/2.0)
    out *= (x[1] <= scene_scale[1]/2.0)*(x[1] >= -scene_scale[1]/2.0)
    return out

def v_given_x_lin(v):
    """
    Takes:
    v: np.array(N_points, 2): points to be evaluated
    Returns probability density of v ~ N(0, sigma_L)
    """
    exponent = - v[0]**2 - v[1]**2
    exponent /= 2*sigma_L**2
    return np.exp(exponent) / (2*np.pi*sigma_L**2)

if __name__ == "__main__":
    pass
    #test measurement_given_x0, ensure it's constant

    print "Actual answer: "
    print x_given_k(np.array([[1, 0], [0,0], [1, 2]]), 0)
    
    print "testing x_given_lin:"
    x = np.zeros((2,3))
    x[0,1] = scene.width/2.0 + 0.1
    x[1,2] = scene.height/2.0 + 0.01
    print "Computed answer = " + str(x_given_lin(x))
    expected_answer = [1.0/(scene.width*scene.height), 0.0, 0.0]
    print "Expected answer = " + str(expected_answer)


    x_span = np.linspace( - scene_scale[0]/2, scene_scale[0]/2, 200)
    y_span = np.linspace( - scene_scale[1]/2, scene_scale[1]/2, 200)
    dvol = (x_span[1]-x_span[0])*(y_span[1]-y_span[0])
    X,Y = np.meshgrid(x_span, y_span)
    x_arr = np.vstack( [X.flatten(), Y.flatten() ] )
    print "\int{P(x|k) dx} =" + str(x_given_k(x_arr,0).sum() * dvol)
    print "Should equal 1"

