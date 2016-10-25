import numpy as np
import itertools
import scipy as sp

def prob_k_s_x0_given_mu(k, s, x0, x, v):
    return np.zeros_like( x0 )

def make_generator(scene, x, v, dt, Nt):
    """ Makes a generator from a scene object and an initial condition
    """
    def gen(k, dt, Nt):
        """ A generator up to time T = Nt * dt
        """
        ode_function = lambda t,x: scene.director_field(k, x)
        from scipy import integrate
        ode_forward = integrate.ode(ode_function)
        ode_forward.t = 0.0
        ode_forward.set_initial_value(initial_condition)
        ode_forward.set_integrator('dopri5')
        ode_backward = integrate.ode(ode_function)
        ode_backward.t = 0.0
        ode_backward.set_initial_value(initial_condition)
        ode_backward.set_integrator('dopri5')
        n = 0
        x_arr = np.zeros((2*Nt+1, initial_condition.size))
        p_arr = np.zeros((2*Nt+1, initial_condition.shape[0]))
        x_arr[0] = initial_condition.flatten()

        ds = dt*scene.s_max
        for n in range(Nt):
            #Compute locations
            ode_forward.integrate(ode_forward.t + dt*scene.s_max)
            assert(ode_forward.successful())
            x_arr[n] = ode_forward.y
            ode_backward.integrate(ode_backward.t - dt*scene.s_max)
            assert(ode_backward.successful())
            x_arr[-n] = ode_backward.y

            #Computes weights
            for l in range(-n, n+1):
                s_l = l*scene.s_max / float(n)
                p_arr[l] = prob_k_s_x0_given_mu(k, s_l, x_arr[l], x, v)
            #TODO:  As coded, you are leaving it up to user to ignor the 0-indices
            yield x_arr, p_arr

    #TODO:  YOU ARE STILL NOT INCLUDING THE LINEAR DISTIRBUTION
    def gen_linear(dt, Nt):
        #use scene.sigma_v
        for n in range(Nt):
            yield None  

    return itertools.izip(
            *[ gen(k, dt, Nt) for k in range(scene.num_nl_classes)] )

if __name__ == '__main__':
    import pickle
    from scene import Scene
    with open('test_scene.pkl', 'rb') as f:
        scene = pickle.load(f)
    initial_condition = np.zeros(2)
    dt = 0.1
    Nt = 10
    x, v = np.random.randn(2)
    gen = make_generator(scene, x, v, dt, Nt)
    for data in gen:
        for x,p in data:
            print x
