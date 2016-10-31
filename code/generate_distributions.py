import numpy as np
from scipy import integrate
import itertools
import scipy as sp
from scene import Scene

from derived_posteriors import prob_k_s_x0_given_mu

def make_generator(scene, x, v, dt, Nt):
    """ Makes a generator from a scene object and an initial condition
    
    args:
        scene: a scene object
        x: np.array(2)
        v: np.array(2)
        dt: float
        Nt: int
   
    returns:
        a list of generators [g_0, ..., g_K]
        where g_0.next() outputs two arrays, x, and w
        where x is a np.array(2,N_points), w is a np.array(N_points)
    """
    
    def gen(k):
        """ A generator up to time T = Nt * dt"""

        def ode_function(t, state):
            xy = state.reshape((2,len(state)/2))
            return scene.director_field_vectorized(k,xy).flatten()
        ode_forward = integrate.ode(ode_function)
        ode_forward.t = 0.0
        x_span = np.linspace(
                x[0]-scene.bbox_width/2.0,
                x[0]+scene.bbox_width/2.0,
                10)
        y_span = np.linspace(
                x[1]-scene.bbox_width/2.0,
                x[1]+scene.bbox_width/2.0,
                10)
        xy = np.vstack(
                [spam.flatten() for spam in np.meshgrid(x_span, y_span)])
        N_points = xy.shape[1]
        initial_condition = xy.flatten()
        ode_forward.set_initial_value(initial_condition)
        ode_forward.set_integrator('dopri5')
        ode_backward = integrate.ode(ode_function)
        ode_backward.t = 0.0
        ode_backward.set_initial_value(initial_condition)
        ode_backward.set_integrator('dopri5')
        xy_arr = np.zeros((2*Nt+1, 2, N_points))
        weight_arr = np.zeros((2*Nt+1, N_points))
        xy_arr[0] = initial_condition.reshape((2, N_points))
        for n in range(1,Nt):
            ds = scene.s_max / float(n)
            #Compute locations
            ode_forward.integrate(ode_forward.t + dt*scene.s_max)
            assert(ode_forward.successful())
            xy_arr[n] = ode_forward.y.reshape(2, N_points)
            ode_backward.integrate(ode_backward.t - dt*scene.s_max)
            assert(ode_backward.successful())
            xy_arr[-n] = ode_backward.y.reshape(2, N_points)

            #Computes weights
            for l in range(-n, n+1):
                x_l = xy_arr[l]
                s_l = l*ds*np.ones( x_l.shape[1] )
                weight_arr[l] = ds*prob_k_s_x0_given_mu(
                        k, s_l, np.transpose(x_l), x, v)
            #x_out = np.concatenate([xy_arr[-n:], xy_arr[:n+1]], axis=0)
            #weight_out = np.concatenate([weight_arr[-n:], weight_arr[:n+1]],
            #        axis=0)
            yield xy_arr, weight_arr

    #TODO:  YOU ARE STILL NOT INCLUDING THE LINEAR DISTIRBUTION
    def gen_linear():
        #use scene.sigma_v
        for n in range(Nt):
            yield None  

    return itertools.izip(
            *[gen(k) for k in range(scene.num_nl_classes)])

if __name__ == '__main__':
    import pickle
    from scene import Scene
    with open('test_scene.pkl', 'rb') as f:
        scene = pickle.load(f)
    with open('test_set.pkl', 'rb') as f:
        test_set = pickle.load(f)
    dt = 0.1
    Nt = 10
    test_BB_ts = test_set[3]
    from matplotlib import pyplot as plt
    plt.plot( test_BB_ts[0], test_BB_ts[1] )
    plt.axis( [-scene.width, scene.width, -scene.height, scene.height] )
    plt.axis('equal')
    plt.show()
    def get_initial_condition(BB_ts):
        fd_width = 4
        BB0 = BB_ts[:,0]
        BB2 = BB_ts[::,,fd_width]
        x = 0.5*(BB0[0]+BB0[2]+BB2[0]+BB2[2]) / fd_width
        y = 0.5*(BB0[1]+BB0[3]+BB2[1]+BB2[3]) / fd_width
        u = 0.5*(BB2[0]-BB0[0]+BB2[2]-BB0[2]) / fd_width
        v = 0.5*(BB2[1]-BB0[1]+BB2[3]-BB0[3]) / fd_width
        return np.array([x, y]), np.array([u, v])

    x,v = get_initial_condition(test_BB_ts[:, 10:])
    speed = np.sqrt(np.sum(v**2))
    print "Measured speed / sigma_v = {:f}".format( speed / scene.sigma_v )
    print "sigma_v = {:f}".format( scene.sigma_v)
    gen = make_generator(scene, x, v, dt, Nt)
    for data in gen:
        for xy,p in data:
            print p.max()
