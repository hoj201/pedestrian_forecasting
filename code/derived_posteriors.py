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
sigma_x = scene.bbox_width / 4.0
vel_width = np.ones([2]) * scene.bbox_velocity_width
sigma_v = 2*sigma_x
sigma_L = scene.sigma_L
p_of_lin = scene.P_of_c[-1]
scene_scale = np.array([scene.width, scene.height])

def joint_k_s_x_x_hat_v_hat(k, s, x, x_hat, v_hat):
    """
    returns the joint probability of k,x,x_hat,v_hat
    
    args:
    k: int
    s: float
    x: numpy.ndarray, shape=(2,N)
    x_hat: numpy.ndarray, shape=(2,)
    v_hat: numpy.ndarray, shape=(2,)
    """
    r2 = (x[0]-x_hat[0])**2 + (x[1]-x_hat[1])**2
    out = np.exp( -r2/(2*sigma_x**2) ) / (2*np.pi*sigma_x**2) #P(x_hat|x)
    v = s*scene.director_field_vectorized(k, x)
    r2 = (v[0]-v_hat[0])**2 + (v[1]-v_hat[1])**2
    out *= np.exp( -r2/(2*sigma_v**2) ) / (2*np.pi*sigma_v**2) #P(v_hat|v)
    out *= posteriors.x_given_k(x,k)
    out *= scene.P_of_c[k]
    out *= 1.0/(2*s_max) * (s <= s_max) * (s >= -s_max) #P(s)
    return out

def joint_lin_x_x_hat_v_hat(x, x_hat, v_hat):
    """
    returns the joint probability P(lin,x,x_hat,v_hat)
    Takes:
    x: np.array(2,N): points to be evaluated
    x_hat: np.array(2): position measurement
    v_hat: np.array(2): velocity measurement
    """
    r2 = (x[0]-x_hat[0])**2 + (x[1]-x_hat[1])**2
    out = np.exp( -r2/(2*sigma_x**2) ) / (2*np.pi*sigma_x**2) #P(x_hat|x)
    out *= posteriors.x_given_lin(x)
    out *= scene.P_of_c[-1]
    v_hat2 = v_hat[0]**2 + v_hat[1]**2
    out *= np.exp( -v_hat2 / (2*(sigma_L**2 + sigma_v**2)))
    out /= 2*np.pi * (sigma_v**2 + sigma_L**2) #int P(\hat{v}|v) P(v|Lin) dv
    return out

def joint_lin_x_t_x_hat_v_hat(t, x_t, x_hat, v_hat):
    """
    returns the joint probabiity P(lin, x_t, x_hat, v_hat)
    Takes:
    t: float: current time
    x_t: np.array(2,N): points to be evaluated
    x_hat: np.array(2): position measurement
    v_hat: np.array(2): velocity measurement
    """
    v_hat2 = np.dot(v_hat,v_hat)
    x_hat2 = np.dot(x_hat,x_hat)
    width = scene.width
    height = scene.height
    area = width*height
    A = scene.P_of_c[-1]
    A /= area * 8 * np.pi**3 * sigma_L**2 * sigma_x**2 * sigma_v**2
    a = -t**2 / (2*sigma_x**2) - (2*sigma_v**2)**-1 - (2*sigma_L**2)**-1
    h = np.zeros(x_t.shape)
    k = np.zeros(x_t.shape)
    for i in (0,1):
        h[i] = (v_hat[i] * sigma_x**2 + (x_hat[i] - x_t[i])*t*sigma_v**2)
        h[i] /= sigma_v**2 * t**2 + sigma_x**2 + sigma_v**2 * sigma_x**2 / sigma_L**2
        k[i] = -v_hat2 * sigma_L**2 * t**2 / 2 - v_hat2 * sigma_x**2 / 2
        k[i] += v_hat[i] * (x_hat[i] - x_t[i]) * sigma_L**2 * t
        k[i] -= x_hat2 * (sigma_L**2-sigma_x**2) / 2
        k[i] += x_hat[i]*x_t[i] * (sigma_L**2 + sigma_v**2)
        k[i] -= x_t[i]**2 * (sigma_L**2 + sigma_v**2) / 2
        k[i] /= sigma_L**2 * sigma_v**2 * t**2 + sigma_L**2 * sigma_x**2 + sigma_x**2 * sigma_v**2
    from scipy.special import erf
    def anti_derivative(u,i):
        arg = u - h[i]
        out = np.exp(k[i])*np.pi*erf(np.sqrt(-a)*(arg))
        out /= (2*np.sqrt(-a)) 
        return out
    u_min = (x_t[0] - width/2) / t
    u_max = (x_t[0] + width/2) / t
    v_min = (x_t[1] - height/2) / t
    v_max = (x_t[1] + height/2) / t
    I0 = anti_derivative(u_max,0) - anti_derivative(u_min,0)
    I1 = anti_derivative(u_max,1) - anti_derivative(u_min,1)
    return A * I0 * I1

if __name__ == "__main__":
    print "Test: \int P(k,s,x,\hat{x},\hat{v}) d\hat{v} = P(k,s,x,\hat{x})."
    k = 0
    s = np.random.rand()*s_max

    #FIND A POINT WHERE P(x|k) is large
    x_arr = np.zeros((2,40))
    x_arr[0,:20] = np.linspace(-scene.width/2, scene.width/2, 20)
    x_arr[1,20:] = np.linspace(-scene.height/2,scene.height/2, 20)
    store = posteriors.x_given_k(x_arr,k)
    i_max = store.argmax()
    x = x_arr[:,i_max]
    x_hat = x + sigma_x*np.random.randn(2)
    v = s*scene.director_field_vectorized(k,x)

    #Now we integrate over v_hat
    u_arr = np.linspace( v[0]-5*sigma_v, v[0]+5*sigma_v, 100)
    v_arr = np.linspace( v[1]-5*sigma_v, v[1]+5*sigma_v, 100)
    dv_hat = (u_arr[1]-u_arr[0])*(v_arr[1]-v_arr[0])
    Q = 0
    from itertools import product
    for u,v in product(u_arr, v_arr):
        v_hat = np.array([u,v])
        Q += joint_k_s_x_x_hat_v_hat(k,s,x,x_hat,v_hat)*dv_hat

    print "computed answer = " + str(Q)
    answer = scene.P_of_c[k] / (2*s_max) * posteriors.x_given_k(x,k)
    from scipy.stats import multivariate_normal
    answer *= multivariate_normal.pdf(x_hat, mean=x, cov=sigma_x**2)
    print "expected answer = " + str(answer)

    print "Test: \int P(Lin,x,\hat{x},\hat{v}) d\hat{v} = P(Lin,x,\hat{x})"
    u_arr = np.linspace( -5*sigma_v, 5*sigma_v, 100)
    v_arr = np.linspace( -5*sigma_v, 5*sigma_v, 100)
    dv_hat = (u_arr[1]-u_arr[0])*(v_arr[1]-v_arr[0])
    Q = 0
    for u,v in product(u_arr, v_arr):
        v_hat = np.array([u,v])
        Q += joint_lin_x_x_hat_v_hat(x,x_hat,v_hat)*dv_hat

    print "computed answer = " + str(Q)
    answer = scene.P_of_c[-1] * (scene.width * scene.height)**-1
    answer *= multivariate_normal.pdf(x_hat, mean=x, cov=sigma_x**2)
    print "expected answer = " + str(answer)

    t = 100.0
    x_hat = np.zeros(2)
    v_hat = np.ones(2)*sigma_v/2
    x_span = np.linspace(-scene.width/2, scene.width/2, 20)
    y_span = np.linspace(-scene.height/2, scene.height/2, 20)
    X,Y = np.meshgrid(x_span, y_span)
    x_t  = np.vstack([X.flatten(), Y.flatten()])
    Z = joint_lin_x_t_x_hat_v_hat(t, x_t, x_hat, v_hat).reshape( X.shape)
    from matplotlib import pyplot as plt
    plt.contourf(X,Y,Z,30)
    plt.show()

    print "Test: \lim_{t \to 0} P(Lin, x_t, \hat{x_0}, \hat{v_0}) = P(Lin, x_0, \hat{x_0}, \hat{v_0})"

    x_xs = np.linspace(-scene.width/2, scene.width/2, 100)
    x_ys = np.linspace(-scene.height/2, scene.height/2, 100)
    xs, ys = np.meshgrid(x_xs, x_ys)
    xs = xs.flatten()
    ys = ys.flatten()
    pts = np.array([xs, ys])
    x = np.array([0, 0])
    v = np.array([0, 2*sigma_v])

    print "The following series depicts approaching t=0 from above\nand how it differs from at x_0\n\nThe average difference should approach zero"
    t = 100.0
    avg = lambda c, v: np.average(np.abs(c-v))
    control = joint_lin_x_x_hat_v_hat(pts, x, v)
    for i in range(9):
        print "T = {}:".format(t)
        vals = joint_lin_x_t_x_hat_v_hat(t, pts, x, v)
        print "Average difference from x_0: {}".format(avg(control, vals))
        t /= 10.0

    print "Test \int P(Lin, x_t, \hat{x_0}, \hat{v_0})\,dx_t = \int P(Lin, x_0, \hat{x_0}, \hat{v_0})\,dx_0"

    bounds = [-scene.width/2, scene.width/2, -scene.height/2, scene.height/2]

    print "All of the following should be equal"

    t = 100.0
    avg = lambda c, v: np.average(np.abs(v-c))
    def temp(xs, ys):
        pts = np.array([xs.flatten(), ys.flatten()])
        return joint_lin_x_x_hat_v_hat(pts, x, v)
    print "x_0: {}".format(trap_quad(temp, bounds, res=(100, 100)))
    for i in range(9):
        def temp(xs, ys):
            pts = np.array([xs.flatten(), ys.flatten()])
            return joint_lin_x_t_x_hat_v_hat(t, pts, x, v)
        print "T={}: {}".format(t, trap_quad(temp, bounds, res=(100, 100)))
        t /= 10.0





