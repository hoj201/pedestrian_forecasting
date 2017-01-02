import numpy as np
from scipy import integrate
import itertools
import scipy as sp
from scene import Scene
from functools import partial
import matplotlib.pyplot as plt

from derived_posteriors import joint_lin_x_x_hat_v_hat
from derived_posteriors import joint_k_s_x_x_hat_v_hat
from derived_posteriors import joint_lin_x_t_x_hat_v_hat

def particle_generator(scene, x_hat, v_hat, t_final, dt=1.0):
    """
    returns a generator which gives particles and weights
    Takes:
    x_hat: np.array(2): position measurement
    v_hat: np.array(2): velocity measurement
    t_final: float
    dt: float

    Example:
    for x,w in particle_generator( scene, x_hat, v_hat, 3.0):
        plt.scatter(x[0], x[1]) #NOTE: This plots the points
        print w.sum() #This prints the total mass
    """
    num_nl_classes = len(scene.P_of_c)-1
    for k in range(num_nl_classes)
        #TODO: Use ODEINT TO INTEGRATE FORWARD AND BACKWARDS IN TIME FOR EACH k
        #TODO: Store results in x_arr, w_arr
    
    for x,w in zip(x_arr,w_arr):
        yield x,w
    pass

if __name__ == '__main__':
    import pickle
    from scene import Scene
    from integrate import trap_quad
    import matplotlib.pyplot as plt
    with open('test_scene.pkl', 'rs') as f:
        scene = pickle.load(f)
    with open('test_set.pkl', 'rs') as f:
        test_set = pickle.load(f)
    dt = 1.0
    Nt = 60
    test_BB_ts = test_set[3]
    def get_initial_condition(BB_ts):
        fd_width = 4
        BB0 = BB_ts[:,0]
        BB2 = BB_ts[::,fd_width]
        x = 0.5*(BB0[0]+BB0[2]+BB2[0]+BB2[2]) / fd_width
        y = 0.5*(BB0[1]+BB0[3]+BB2[1]+BB2[3]) / fd_width
        u = 0.5*(BB2[0]-BB0[0]+BB2[2]-BB0[2]) / fd_width
        v = 0.5*(BB2[1]-BB0[1]+BB2[3]-BB0[3]) / fd_width
        return np.array([x, y]), np.array([u, v])

    x,v = get_initial_condition(test_BB_ts[:, 10:])
    plt.scatter(x[0], x[1], s=60)
    speed = np.sqrt(np.sum(v**2))
    print "Measured speed / sigma_v = {:f}".format( speed / scene.sigma_v )
    print "sigma_v = {:f}".format( scene.sigma_v)
