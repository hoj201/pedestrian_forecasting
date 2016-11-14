import numpy as np
#from energy import Vk
from numpy.polynomial import legendre
from integrate import trap_quad
from scipy.stats import multivariate_normal
import pickle
from scene import Scene


if __name__ == "__main__":
    Vk = np.array([[[0, 0], [1,0]]])
    scene_scale = np.ones([2])
    dist_width = np.ones([2])
    vel_width = np.ones([2])
    s_max = 1
else:
    with open("test_scene.pkl", "rb") as f:
        scene = pickle.load(f)
    Vk = scene.alpha_arr
    scene_scale = np.array([scene.width, scene.height])
    #temporary
    dist_width = np.ones([2]) * scene.bbox_width
    vel_width = np.ones([2]) * scene.bbox_velocity_width
    s_max = scene.s_max

def legval_scale(x, y, coeffs):
    """
    Returns the values of a legend repolynomial
    at N points, scaled so the whole plane is [0,1]**2

    Takes x: np.array(N_Points): x values to evaluate
    Takes y: np.array(N_Points): y values to evaluate
    Takes coeffs: np.array(K_max_degree): legendre polynomial coefficients

    Returns np.array(N_Points)

    """
    return legendre.legval2d(x/scene_scale[0], y/scene_scale[1], coeffs)



def measurement_given_x0(x, x0):
    """
    Returns the probability at x given an x0

    Takes x0: np.array(2): initial point
    Takes x: np.array(N_points, 2): set of points at which to evaluate probability

    Returns np.array(N_Points): probability that agent is at x given x0

    """
    #function to determine if points are within bounding box
    ident = np.logical_and(np.absolute(x[:, 0] - x0[0]) <= dist_width[0]/2.0, np.absolute(x[:, 1] - x0[1]) <= dist_width[0]/2.0)
    return 1.0/( dist_width[0]**2) * ident


def measurement_given_v0(v, v0):
    """
    Returns the probability at points v given a v0

    Takes v0: np.array(2): initial point
    Takes v: np.array(N_points, 2): set of points at which to evaluate probability

    Returns np.array(N_Points): probability that agent is at v given v0
    """
    #filters out points not inside the bounding box.
    ident = np.logical_and(np.absolute(v[:, 0] - v0[0]) <= vel_width[0]/2, np.absolute(v[:, 1] - v0[1]) <= vel_width[0]/2)
    return 1.0/(vel_width[0]**2) * ident

def prob_s_uniform(s):
    """
    Returns P(s) assuming a uniform distribution of +-s_max
    s_max: float
    s: np.array(n_points): s values to be sampled
    """
    ident = s <= s_max
    return 1.0/(2*s_max) * ident


_normalization_constants = []
_bounds = (-1 * scene_scale[0], scene_scale[0], -1 * scene_scale[1], scene_scale[1])

for coeffs in Vk:
    fn = lambda x, y: np.exp( -1*legval_scale(x, y, coeffs))
    _normalization_constants.append(trap_quad(fn, _bounds))


def x0_given_k(k, x0):
    """
    Returns probability that an agent will be at x0 given k

    Takes k: int: cluster
    Takes x0: np.array(N_points, 2)

    Returns np.array(N_points)
    According to the equation (1/Z_K)exp(-1*V_k)
    """
    return (1.0/_normalization_constants[k])*np.exp(-1 * legval_scale(x0[:, 0], x0[:, 1], Vk[k]))

def x0_given_lin(x0):
    """
    takes x0: np.array(N_points, 2): points to be evaluated

    Returns np.array(N_points)
    """
    return 1/(scene_scale[0] * scene_scale[1]) * np.ones(len(x0))

def v0_given_x0_lin(v0):
    """
    Takes:
    v0: np.array(N_points, 2): points to be evaluated
    Returns probability of v0 := N(0, sigma_v)
    """
    return multivariate_normal.pdf(v0, np.zeros([2]), scene.sigma_v**2)

if __name__ == "__main__":
    pass
    #test measurement_given_x0, ensure it's constant

    assert measurement_given_x0(np.array([[0,0]]), np.array([0,0]))[0] == measurement_given_x0(np.array([[0,0]]), np.array([0,0.5]))[0]

    print "Expected answer: 0.07825"

    print "Actual answer: "

    print x0_given_k(0, np.array([[1, 0], [0,0], [1, 2]]))

    print "testing x0_given_lin"
    print x0_given_lin([[0,0],[0.5,0.5], [2,2]])

    print measurement_given_v0([[0,0], [0.5, 0.5], [1,1], [2,2]], [0,0])

    


