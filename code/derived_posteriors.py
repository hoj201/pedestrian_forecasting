from scene import Scene
import posteriors
import pickle
import numpy as np
from integrate import trap_quad
from scipy.special import erf

with open("test_scene.pkl", "rb") as f:
    scene = pickle.load(f)
max_k = len(scene.P_of_c)-1
#temporary s_max
s_max = 1
s_max = scene.s_max
#temporary dist_width
dist_width = np.ones([2]) * scene.bbox_width
vel_width = np.ones([2]) * scene.bbox_velocity_width
sigma_v = scene.sigma_v
p_of_lin = scene.P_of_c[-1]
scene_scale = np.array([scene.width, scene.height])


def _prob_k_s_x0(k, s, x0, x, v):
    """
    returns P(k,s,x0 | mu)
    Takes:
    k: float: class of agent
    x0: np.array(N_points, 2): points to be evaluated
    x: np.array(2): given measurement of x0
    v: np.array(2): given measurement of v0
    s: np.array(N_points): initial speed
    """
    prob_measurement = posteriors.measurement_given_x0(x0, x)

    xs = [x[0] for x in x0]
    ys = [x[1] for x in x0]
    xy = np.array([xs, ys])
    initial_velocity = scene.director_field_vectorized(k, xy)
    initial_velocity = np.dstack([s, s])[0] * np.dstack(initial_velocity)[0]
    measurement_given_velocity = posteriors.measurement_given_v0(initial_velocity, v)
    x0_given_k = posteriors.x0_given_k(k, x0)
    p_k = scene.P_of_c[k]
    p_s = posteriors.prob_s_uniform(s)
    ret = prob_measurement * measurement_given_velocity * p_k * p_s * x0_given_k
    return ret

def _prob_lin_x_mu(x0, x, v):
    """
    returns P(lin,x|mu)
    Takes:
    x0: np.array(N_points, 2): points to be evaluated
    x: np.array(2): given measurement of x0
    v: np.array(2): given measurement of v0
    """
    product = 1
    product *= posteriors.measurement_given_x0(x0, x) * scene.P_of_c[-1] * posteriors.x0_given_lin(x0)
    product /= 4 *  vel_width[0]**2 #* scene_scale[0] * scene_scale[1]
    term1 = erf((v[0] + vel_width[0]/2)/(np.sqrt(2) * sigma_v))
    term2 = erf((v[0] - vel_width[0]/2)/(np.sqrt(2) * sigma_v))
    product *= term1 - term2
    #problem is in large params to erf
    term1 = erf((v[1] + vel_width[0]/2)/(np.sqrt(2) * sigma_v))
    term2 = erf((v[1] - vel_width[0]/2)/(np.sqrt(2) * sigma_v))
    product *= term1-term2
    return product


x_s = np.zeros([2])
v_s = np.zeros([2])
normalizing_constant_s = None
def _normalizing_constant(x,v):
    global x_s
    global v_s
    global normalizing_constant_s

    res = 40

    if np.array_equal(x,x_s) and np.array_equal(v, v_s) and normalizing_constant_s != None:
        return normalizing_constant_s
    
    bounds = (x[0] - dist_width[0]/2, x[0] + dist_width[0]/2,
              x[1] - dist_width[0]/2, x[1] + dist_width[0]/2,
              -0.5 * s_max, 0.5 * s_max)

    normalizing_constant = 0
    for i in range(max_k):
        def temp(x1,y1,z):
            x1 = x1.flatten()
            y1 = y1.flatten()
            #may be slow, possibly use dstack?
            xy = np.array(zip(x1,y1))
            z = z.flatten()
            return _prob_k_s_x0(i, z, xy, x, v)
        normalizing_constant += trap_quad(temp, bounds, res = (res, res, res))
    bounds = (x[0] - dist_width[0]/2, x[0] + dist_width[0]/2,
              x[1] - dist_width[0]/2, x[1] + dist_width[0]/2)
              #v[0] - vel_width[0]/2, v[0] + vel_width[0]/2,
              #v[1] - vel_width[1]/2, v[1] + vel_width[1]/2)
    def temp(x1,y1):
        x1 = x1.flatten()
        y1 = y1.flatten()
        #z = z.flatten()
        #w = w.flatten()
        #may be slow, possibly use dstack?
        xy = np.array(zip(x1,y1))
        #zw = np.array(zip(z, w))
        return _prob_lin_x_mu( xy, x, v)
    normalizing_constant += trap_quad(temp, bounds, res = (res, res))
    normalizing_constant_s = normalizing_constant
    x_s = x
    v_s = v
    return normalizing_constant

#placeholder
#meant to be summation over all k, s, x0
#-smax to smax
#all k
#all x within bounding box of x0
def prob_k_s_x0_given_mu(k, s, x0, x, v):
    """
    Takes
    k: int: cluster
    s: np.array(N_points), speeds of each initial point
    x0: np.array(N_points, 2), initial points to be evaluated
    x: np.array(2): given position measurement
    v: np.array(2): given velocity measurement
    """

    return 1.0/_normalizing_constant(x, v) * _prob_k_s_x0(k, s, x0, x, v)

def _prob_lin_x_v_mu(x0, v0, x, v):
    """
    Takes:
    x0: np.array(N_points, 2): positions to be evaluated
    v0: np.array(N_points, 2): velocities to be evaluated
    x: np.array(2): position measurement
    v: np.array(2): velocity measurement
    returns np.array(N_points): P(x0, v0, lin, x, v)
    """
    product = 1
    product *= posteriors.x0_given_lin(x0)
    product *= posteriors.measurement_given_x0(x0, x)
    product *= posteriors.measurement_given_v0(v0, v)
    product *= posteriors.v0_given_x0_lin(v0)
    product *= p_of_lin
    return product

def prob_lin_x_v_given_mu(x0, v0, x, v):
    """
    Takes:
    x0: np.array(N_points, 2): points to be evaluated
    v0: np.array(N_points, 2): velocities to be evaluated
    x: np.array(2): measured position
    v: np.array(2): measured velocity
    Returns np.array(N_points): P(x0, v0, lin|x, v)
    """
    return _prob_lin_x_v_mu(x0, v0, x, v) / _normalizing_constant(x, v)

if __name__ == "__main__":

    sum_k = 0
    x = np.array([0, 0])
    v = np.array([0.0,0])

    print "Starting sanity check"
    print "should be same:"
    x0 = [[0.01, 0.01]]
    print _prob_lin_x_mu(x0, x, v)
    bounds = [v[0] - vel_width[0]/2,  v[0] + vel_width[0]/2,
               v[1] - vel_width[0]/2, v[1] + vel_width[0]/2]
    def temp(x1, y1):
        x1 = x1.flatten()
        y1 = y1.flatten()
        xy = np.array(zip(x1, y1))
        xs = np.tile(x0, (len(xy), 1))
        return _prob_lin_x_v_mu(xs, xy, x, v)
    print trap_quad(temp, bounds)

    print "starting k normalization"
    bounds = (x[0] - dist_width[0]/2, x[0] + dist_width[0]/2,
             x[1] - dist_width[0]/2, x[1] + dist_width[0]/2,
             -0.5 * s_max, 0.5 * s_max)
    for i in range(max_k):
       print i
       def temp(x1,y1,z):
           x1 = x1.flatten()
           y1 = y1.flatten()
           #may be slow, possibly use dstack?
           xy = np.array(zip(x1,y1))
           z = z.flatten()
           return prob_k_s_x0_given_mu(i, z, xy, x, v)
       sum_k += trap_quad(temp, bounds, (40, 40, 40))
    print "starting lin"
    bounds = (x[0] - dist_width[0]/2, x[0] + dist_width[0]/2,
              x[1] - dist_width[0]/2, x[1] + dist_width[0]/2,
              v[0] - vel_width[0]/2, v[0] + vel_width[0]/2,
              v[1] - vel_width[1]/2, v[1] + vel_width[1]/2
              )
    def temp(x1,y1, z, w):
        x1 = x1.flatten()
        y1 = y1.flatten()
        z = z.flatten()
        w = w.flatten()
        xy = np.array(zip(x1,y1))
        zw = np.array(zip(z, w))
        return prob_lin_x_v_given_mu(xy, zw, x, v)
    lin_term = trap_quad(temp, bounds, (40, 40, 40, 40))
    sum_k += lin_term
    print "Should be 1.0:"
    print sum_k

    print "should be close to 1:"
    #print prob_k_s_x0_given_mu(0, [0], x0, x, v)
    def temp(x, y):
        x = x.flatten()
        y = y.flatten()
        #may be slow, possibly use dstack?
        xy = np.array(zip(x,y))
        return posteriors.measurement_given_v0(xy, np.array([0,0]))
    bounds = (-1 * vel_width[0], 1 * vel_width[0], -1 * vel_width[0], 1 * vel_width[0])
    print trap_quad(temp, bounds, res=(40,40))

    print "should be close to 1:"
    bounds = [-1 * s_max, s_max]
    print trap_quad(posteriors.prob_s_uniform, bounds)

    print "all should be close to 1:"
    bounds = []
    for i in range(max_k):
       def temp(x,y):
           x = x.flatten()
           y = y.flatten()
           xy = np.array(zip(x,y))
           return posteriors.x0_given_k(i, xy)
       width = posteriors.scene_scale[0]
       height = posteriors.scene_scale[1]
       bounds = [-2*width/2.0,2* width/2.0, -2*height/2.0, 2*height/2.0]
       print trap_quad(temp, bounds)

    print "should be close to 1:"
    bounds = 3*scene.sigma_v*np.array([-1., 1., -1., 1.])
    bounds = list(bounds)
    def temp(x,y):
       x = x.flatten()
       y = y.flatten()
       xy = np.array(zip(x,y))
       return posteriors.v0_given_x0_lin(xy)
    print trap_quad(temp, bounds, (100,100))

    k = 0
    x0 = np.array([[0,0]])
    x = np.array([0,0])
    v = np.array([0,0])
    s = np.array([-.002])
    dx = np.array([0,0.02])
    print "Should be the same:"
    print _prob_k_s_x0(k, s, x0, x, v) / _prob_k_s_x0(k, s, x0 + dx, x, v)
    print posteriors.x0_given_k(k, x0) / posteriors.x0_given_k(k, x0+dx)

    x = np.array([0,0])
    v = np.array([.005,0.005])
    _normalizing_constant(x, v)

    pass
