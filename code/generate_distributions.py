import numpy as np
import pickle
from scene import Scene
from scipy import integrate
import itertools
import scipy as sp
from functools import partial
import matplotlib.pyplot as plt

import posteriors
from derived_posteriors import joint_lin_x_x_hat_v_hat
from derived_posteriors import joint_k_s_x_x_hat_v_hat
from derived_posteriors import joint_lin_x_t_x_hat_v_hat

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

def integrate_class(k, x0, T, N_steps):
    """
    integrates a director field over a range [-T,T]
    args:
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
        return s_max * scene.director_field_vectorized(k,x).flatten()

    # Integrate the ODE backwards in time
    f_backwards = lambda x,t: -1*f(x,t)
    t_arr = np.linspace(0, T, N_steps+1)
    x_backward = odeint(f_backwards, x0.flatten() , t_arr)[::-1]
    x_forward = odeint(f, x0.flatten() , t_arr)
    x_arr = np.concatenate([x_forward, x_backward[1:]])
    return x_arr.reshape( (2*N_steps+1, 2, N) )

def particle_generator(x_hat, v_hat, t_final, N_steps):
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
    
    #Initializes particles for nonlinear classes
    x_span = np.linspace( - 3*sigma_x, 3*sigma_x, 20)
    dvol_nl = (x_span[1]-x_span[0])**2
    X,Y = np.meshgrid(x_span + x_hat[0], x_span + x_hat[1])
    x0 = np.vstack([X.flatten(), Y.flatten()])
    
    #Initializes a regular grid for evaluation of the linear class
    x_span = np.linspace( -scene.width/2, scene.width/2, 30)
    dx = x_span[1] - x_span[0]
    y_span = np.linspace( -scene.height/2, scene.height/2, 30)
    dy = y_span[1] - y_span[0]
    X,Y = np.meshgrid(x_span, y_span)
    x_lin = np.vstack( [X.flatten(), Y.flatten()])

    N_ptcl = x0.shape[1]
    x_arr = np.zeros((num_nl_classes, 2*N_steps+1, 2, N_ptcl))
    for k in range(num_nl_classes):
        x_arr[k] = integrate_class(k, x0, t_final, N_steps)
    #At time t=0, rho(x,t=0) is nothing but P(x0 | x0_hat),
    # which equals P(x0_hat | x0).
    w_arr = posteriors.x_hat_given_x(x0, x_hat)*dvol_nl
    w_out = w_arr.flatten()
    x_out = x0
    yield x_out, w_out
    #For later times, the class of the agent matters.
    for n in range(1,N_steps):
        #The following computations handle the nonlinear classes
        t = n * t_final / float(N_steps)
        ds = s_max / n 
        w_arr = np.zeros((num_nl_classes, 2*n+1, N_ptcl))
        for k in range(num_nl_classes):
            for m in range(-n,n+1):
                w_arr[k,m] = joint_k_s_x_x_hat_v_hat(
                        k, s_max*m /n, x0, x_hat, v_hat) #TODO: Memoize??
                w_arr[k,m] *= ds * dvol_nl
        x_out = np.zeros((num_nl_classes,2*n+1,2,N_ptcl))
        x_out[:,-n:,:,:] = x_arr[:,-n:,:,:]
        x_out[:,:n+1,:,:] = x_arr[:,:n+1,:,:]
        w_out = w_arr.flatten()
        x_out = np.vstack([x_out[:,:,0,:].flatten(), x_out[:,:,1,:].flatten()])

        #The following computations handle the linear predictor class
        w_lin = joint_lin_x_t_x_hat_v_hat(t, x_lin, x_hat, v_hat) * dy*dx
        #TODO: append regular grid and weights to x_out, w_out
        x_out = np.concatenate( [x_out, x_lin], axis=1)
        w_out = np.concatenate( [w_out, w_lin])
        if n==1:
            prob_of_mu = w_out.sum()
        yield x_out, w_out/ prob_of_mu
    pass
    

def pdf_generator():
    """
    a generator
    """
    pass

if __name__ == '__main__':
    from integrate import trap_quad
    import matplotlib.pyplot as plt
    with open('test_set.pkl', 'rs') as f:
        test_set = pickle.load(f)
    test_BB_ts = test_set[0]

    from process_data import BB_ts_to_curve
    curve = BB_ts_to_curve( test_BB_ts)
    offset = 30
    x_hat = curve[:,offset]
    v_hat = (curve[:,offset+2] - curve[:,offset-2])/4
    print "x_hat = " + str(x_hat)
    print "v_hat = " + str(v_hat)
    speed = np.sqrt(np.sum(v_hat**2))
    print "Measured speed / sigma_L = {:f}".format( speed / scene.sigma_L )
    print "sigma_L = {:f}".format( scene.sigma_L)
    k=0
    N_steps = 60
    t_final = 60
    #Domain is actually larger than the domain we care about
    domain = [-scene.width, scene.width, -scene.height, scene.height]

    gen = particle_generator(x_hat, v_hat, t_final, N_steps)
    n = 0
    from visualization_routines import singular_distribution_to_image
    res = (50,60)
    for x_arr, w_arr in gen:
        if n%5==0:
            X,Y,Z = singular_distribution_to_image(
                    x_arr, w_arr, domain, res=res)
            plt.contourf(X,Y,Z, 20, cmap='viridis')
            plt.plot(curve[0], curve[1])
            plt.scatter( curve[0,offset], curve[1,offset])
            plt.show()
            plt.clf()
        n += 1
