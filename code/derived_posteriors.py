from scene import Scene
import posteriors
import pickle
import numpy as np
from integrate import trap_quad
from scipy.special import erf
import matplotlib.pyplot as plt

from data import scene

max_k = len(scene.P_of_c)-1
#temporary s_max
s_max = 1
s_max = scene.s_max
#temporary dist_width
dist_width = np.ones([2]) * scene.bbox_width
vel_width = np.ones([2]) * scene.bbox_velocity_width
sigma_x = scene.sigma_x
sigma_v = scene.sigma_v
sigma_L = scene.sigma_L
p_of_lin = scene.P_of_c[-1]
scene_scale = np.array([scene.width, scene.height])

def set_scene(num_scene, custom_scene=None):
    global scene, max_k, s_max, dist_width, vel_width, sigma_x, sigma_v, sigma_l, p_of_lin, scene_scale
    if custom_scene!=None:
        scene = custom_scene
    else:
        from data import scenes
        scene = scenes[num_scene]
    max_k = len(scene.P_of_c)-1
    #temporary s_max
    s_max = scene.s_max
    #temporary dist_width
    dist_width = np.ones([2]) * scene.bbox_width
    vel_width = np.ones([2]) * scene.bbox_velocity_width
    sigma_x = scene.sigma_x
    sigma_v = scene.sigma_v
    sigma_L = scene.sigma_L
    p_of_lin = scene.P_of_c[-1]
    scene_scale = np.array([scene.width, scene.height])
    posteriors.set_scene(num_scene, custom_scene=custom_scene)


#"mathematically correct" version
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

#fast version
def joint_k_x_x_hat_v_hat(k, x, x_hat, v_hat):
    """
    returns the joint probability of k,x,x_hat,v_hat
    
    args:
    k: int
    x: numpy.ndarray, shape=(2,N)
    x_hat: numpy.ndarray, shape=(2,)
    v_hat: numpy.ndarray, shape=(2,)
    """
    r2 = (x[0]-x_hat[0])**2 + (x[1]-x_hat[1])**2
    out = np.exp( -r2/(2*sigma_x**2) ) / (2*np.pi*sigma_x**2) #P(x_hat|x)
    #v = s*scene.director_field_vectorized(k, x)
    #r2 = (v[0]-v_hat[0])**2 + (v[1]-v_hat[1])**2
    #out *= np.exp( -r2/(2*sigma_v**2) ) / (2*np.pi*sigma_v**2) #P(v_hat|v)
    out *= posteriors.x_given_k(x,k)
    out *= scene.P_of_c[k]
    #out *= 1.0/(2*s_max) * (s <= s_max) * (s >= -s_max) #P(s)
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
    sqr = lambda x_arr: x_arr[0]**2 + x_arr[1]**2
    N = x_t.shape[1]
    def subtr(a, b):
        a = np.broadcast_to(a, (N,2)).transpose()
        return a - b
    v_hat2 = sqr(v_hat)
    x_hat2 = sqr(x_hat)
    Dx = subtr(x_hat, x_t) #Shape = (2,400)
    width = scene.width
    height = scene.height
    area = width*height
    a = sigma_v**2 * sigma_L**2 * t**2
    a += sigma_L**2 * sigma_x**2
    a += sigma_v**2 * sigma_x**2
    u_off = subtr(v_hat*sigma_x**2, Dx*t*sigma_v**2)
    u_off *= sigma_L**2 / a
    k = -sqr( subtr(v_hat*sigma_x/sigma_v, t*Dx*sigma_v/sigma_x ))
    k *= sigma_L**2 / a
    k += v_hat2 / sigma_v**2 + sqr(Dx) / sigma_x**2
    out = scene.P_of_c[-1] * np.exp( - k / 2.0 )
    
    out /= 16 * np.pi**2 * area * a
    from scipy.special import erf
    def anti_derivative(u,i):
        A = u - u_off[i]
        B = np.sqrt(t**2 * sigma_x**-2 + sigma_v**-2 + sigma_L**-2)
        return erf(A * B)
    u_min = (x_t[0] - width/2) / t
    u_max = (x_t[0] + width/2) / t
    v_min = (x_t[1] - height/2) / t
    v_max = (x_t[1] + height/2) / t
    out *= anti_derivative(u_max,0) - anti_derivative(u_min,0)

    out *= anti_derivative(v_max,1) - anti_derivative(v_min,1)
    return out

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
    #THE FOLLOWING IS TO PLOT
    #x_span = np.linspace(-scene.width/2, scene.width/2, 20)
    #y_span = np.linspace(-scene.height/2, scene.height/2, 20)
    #X,Y = np.meshgrid(x_span, y_span)
    #x_t  = np.vstack([X.flatten(), Y.flatten()])
    #Z = joint_lin_x_t_x_hat_v_hat(t, x_t, x_hat, v_hat).reshape( X.shape)
    #from matplotlib import pyplot as plt
    #plt.contourf(X,Y,Z,30)
    #plt.show()

    print "Test: \lim_{t \to 0} P(Lin, x_t, \hat{x_0}, \hat{v_0}) = P(Lin, x_0, \hat{x_0}, \hat{v_0})"

    x_xs = np.linspace(-scene.width/2, scene.width/2, 100)
    x_ys = np.linspace(-scene.height/2, scene.height/2, 100)
    xs, ys = np.meshgrid(x_xs, x_ys)
    xs = xs.flatten()
    ys = ys.flatten()
    pts = np.array([xs, ys])
    x = np.array([0, 0])
    v = np.array([0, 2*sigma_v])

    print """
    The following series depicts approaching t=0 from above
    and how it differs from at x_0

    The average difference should approach zero"""

    t = 100.0
    avg = lambda c, v: np.average(np.abs(c-v))
    oom = lambda c, v: avg(np.log(c), np.log(v))
    mxoom = lambda c, v: np.amax(np.abs(np.log(c)-np.log(v)))
    mnoom = lambda c, v: np.amin(np.abs(np.log(c)-np.log(v)))
    control = joint_lin_x_x_hat_v_hat(pts, x, v)
    for i in range(9):
        print "T = {}:".format(t)
        vals = joint_lin_x_t_x_hat_v_hat(t, pts, x, v)
        print "Average difference from x_0: {}".format(avg(control, vals))
        t /= 10.0

    print "Test \int P(Lin, x_t, \hat{x_0}, \hat{v_0})\,dx_t = \int P(Lin, x_0, \hat{x_0}, \hat{v_0})\,dx_0"

    #We integrate over a larger region than the domain because much of the mass leaves the domain

    print "All of the following should be equal"
    bounds = np.array([x[0],x[0],x[1],x[1]])
    offset = np.array([-1,1,-1,1])*7.0*sigma_x
    bounds = bounds + offset
    t = 100.0
    res = (1000,1000)
    def temp(xs, ys):
        pts = np.array([xs.flatten(), ys.flatten()])
        return joint_lin_x_x_hat_v_hat(pts, x, v)
    print "x_0: {}".format(trap_quad(temp, bounds, res=res))
    for i in range(6):
        def temp(xs, ys):
            pts = np.array([xs.flatten(), ys.flatten()])
            return joint_lin_x_t_x_hat_v_hat(t, pts, x, v)
        offset_t = np.array([v[0],v[0],v[1],v[1]]) * t
        doffset_t = np.array([-1,1,-1,1]) * 5*sigma_v * t
        offset_t = offset_t + doffset_t
        bounds_t = bounds + offset_t
        print "T={}: {}".format(t, trap_quad(temp, bounds_t, res=res))
        t /= 10.0

    print """
    Test that the closed-form solution for P(x_0, Lin, \mu)
    is the same as a numerically integrated one.
    """

    bounds = [-7*sigma_v, 7*sigma_v,
              -7*sigma_v, 7*sigma_v]
    x = np.array([0, 0])
    v = np.array([0, sigma_v])

    xs = np.linspace(0, scene.width/2, 20)
    ys = np.linspace(0, scene.height/4, 20)
    pts = np.array([xs, ys])
    results = []
    results2 = []

    for i in range(20):
        pt = np.array([pts[0, i], pts[1, i]])
        def temp(xs, ys):
            pts = np.array([xs.flatten(), ys.flatten()])
            res = posteriors.x_given_lin(pt) * scene.P_of_c[-1]
            res *= posteriors.x_hat_given_x(x, pt) * posteriors.v_hat_given_v(v, pts)
            res *= posteriors.v_given_x_lin(pts) 
            return res
        results.append(trap_quad(temp, bounds, res = res))
        results2.append(joint_lin_x_x_hat_v_hat(pt, x, v))
    results = np.array(results)
    results2 = np.array(results2)
    print "Average difference between control and test:"
    print avg(np.array(results), np.array(results2))
