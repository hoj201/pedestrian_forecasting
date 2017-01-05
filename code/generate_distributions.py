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

def integrate_class(scene, k, x0, T, N_steps):
    """
    integrates a director field over a range [-T,T]
    args:
    scene: an instance of Scene
    x0: np.array.shape == (2,N)
    T: float
    N_steps: int

    Returns:
    np.array.shape == (2*N_steps+1, 2, N)
    """
    N = x0.shape[1] #number of particles
    from scipy.integrate import odeint
    def f(x,t):
        x = x.reshape(2,len(x)/2)
        s_max = scene.s_max
        return s_max * scene.director_field_vectorized(k,x).flatten()

    # Integrate the ODE backwards in time
    f_backwards = lambda x,t: -1*f(x,t)
    t_arr = np.linspace(0, T, N_steps+1)
    x_arr1 = odeint(f_backwards, x0.flatten() , t_arr)[::-1]
    x_arr1 = x_arr1.reshape((N_steps+1, 2, N))
    x_arr2 = odeint(f, x0.flatten() , t_arr).reshape((N_steps+1, 2, N))
    return np.concatenate([x_arr2, x_arr1[1:]], axis=0)

def particle_generator(scene, x_hat, v_hat, t_final, N_steps):
    """
    a generator which gives particles and weights
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
    sigma_x = scene.bbox_width / 4.0
    s_max = scene.s_max
    x_span = np.linspace( - sigma_x, sigma_x, 5 )
    X,Y = np.meshgrid(x_span + x_hat[0], x_span + x_hat[1] )
    x0 = np.vstack([X.flatten(), Y.flatten()])
    N_ptcl = x0.shape[1]
    x_arr = np.zeros((num_nl_classes, 2*N_steps+1, 2, N_ptcl))
    for k in range(num_nl_classes):
        x_arr[k] = integrate_class(scene, k, x0, t_final, N_steps)
    #TODO: include the n=0 case
    for n in range(1,N_steps):
        ds = s_max / n 
        w_arr = np.zeros((num_nl_classes, 2*n+1, N_ptcl))
        for k in range(num_nl_classes):
            for m in range(-n,n+1):
                w_arr[k,m] = ds * joint_k_s_x_x_hat_v_hat(k, s_max*m /n, x0, x_hat, v_hat) #TODO: Use memoization here
        w_out = w_arr.flatten()
        x_out = np.zeros((2,N_ptcl*num_nl_classes*(2*n+1)))
        x_out[0,:n*num_nl_classes*N_ptcl] = x_arr[:,2*N_steps+1-n:,0,:].flatten()
        x_out[0,n*N_ptcl*num_nl_classes:] = x_arr[:,:n+1,0,:].flatten()
        x_out[1,:n*num_nl_classes*N_ptcl] = x_arr[:,2*N_steps+1-n:,1,:].flatten()
        x_out[1,n*N_ptcl*num_nl_classes:] = x_arr[:,:n+1,1,:].flatten()
        yield x_out, w_out
    pass
    

def pdf_generator():
    """
    a generator
    """
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

    x_hat, v_hat = get_initial_condition(test_BB_ts[:, 5:])
    print "x_hat = " + str(x_hat)
    print "v_hat = " + str(v_hat)
    speed = np.sqrt(np.sum(v_hat**2))
    print "Measured speed / sigma_L = {:f}".format( speed / scene.sigma_L )
    print "sigma_L = {:f}".format( scene.sigma_L)
    k=0
    N_steps = 5
    t_final = 60
    domain = [-scene.width/2, scene.width/2, -scene.height/2, scene.height/2]

    for x_arr, w_arr in particle_generator(scene, x_hat, v_hat, t_final, N_steps):
        plt.scatter(x_arr[0], x_arr[1], marker='.')
        plt.axis(domain)
        plt.show()
        plt.clf()

