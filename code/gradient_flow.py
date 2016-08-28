import numpy as np
from numpy.polynomial.legendre import legder, legval2d

def predict( theta, s, x0, y0, t_arr ):
    global V_scale
    """ Returns a predicted trajectory

    args:
        theta: potential function parameters ( 2D numpy.ndarray)
        s: time scale (float)
        x0: initial x coordinate (float)
        y0: initial y coordinate (float)
        t_arr: times nodes ( 1D numpy.ndarray )

    returns:
        x_pred: x coordinates at times in t_arr (numpy.ndarray)
        y_pred: x coordinates at times in t_arr (numpy.ndarray)
    """
    def ode_func( xy, t, theta, s ):
        x = xy[0]
        y = xy[1]
        epsilon = 0.1
        dUdx = legval2d( x/V_scale[0], y/V_scale[1], legder(theta, axis=0) ) / V_scale[0]
        dUdy = legval2d( x/V_scale[0], y/V_scale[1], legder(theta, axis=1) ) / V_scale[1]
        mag_dU = np.sqrt( dUdx**2 + dUdy**2 )
        epsilon = 0.1
        out = np.zeros_like(xy)
        out[0] = - s*dUdx / (mag_dU + epsilon)
        out[1] = - s*dUdy / (mag_dU + epsilon)
        return out
    from scipy.integrate import odeint
    xy0 = np.array([x0,y0])
    xy_arr = odeint( ode_func, xy0, t_arr, args=(theta,s) ) 
    return xy_arr[:,0], xy_arr[:,1]


def Ds_predict( theta, s, x0, y0, t_arr ):
    """ Returns the sensitivity of a predicted trajectory to perturbations in s

    args:
        theta: potential function parameters ( 2D numpy.ndarray)
        s: time scale (float)
        x0: initial x coordinate (float)
        y0: initial y coordinate (float)
        t_arr: times nodes ( 1D numpy.ndarray )

    returns:
        x_pred: x coordinates at times in t_arr (numpy.ndarray)
        y_pred: y coordinates at times in t_arr (numpy.ndarray)
        Ds_x_pred: sensitivities of x coordinates at times in t_arr (numpy.ndarray)
        Ds_y_pred: sensitivities of y coordinates at times in t_arr (numpy.ndarray)
    """
    def ode_func( state, t, theta, s ):
        x = state[0]
        y = state[1]
        dxds = state[2]
        dyds = state[3]
        epsilon = 0.1
        dUdx = legval2d( x/V_scale[0], y/V_scale[1], legder(theta, axis=0) ) / V_scale[0]
        dUdy = legval2d( x/V_scale[0], y/V_scale[1], legder(theta, axis=1) ) / V_scale[1]
        mag_dU = np.sqrt( dUdx**2 + dUdy**2 )
        epsilon = 0.1
        out = np.zeros_like( state )
        out[0] = - s*dUdx / (mag_dU + epsilon)
        out[1] = - s*dUdy / (mag_dU + epsilon)
        out[2] = - dUdx / (mag_dU + epsilon)
        out[3] = - dUdy / (mag_dU + epsilon)
        return out
    from scipy.integrate import odeint
    state0 = np.array([ x0, y0, 0.0, 0.0] )
    state_arr = odeint( ode_func, state0, t_arr, args=(theta,s) , rtol=1e-8) 
    return state_arr[:,0], state_arr[:,1], state_arr[:,2], state_arr[:,3]


def Dtheta_grad_U( x , y ):
    global V_scale, k_max
    """ Returns dU_dxdtheta

    args:
        x: float
        y: float

    returns:
        dUdx_dtheta, dUdy_dtheta (numpy.ndarray)
    """
    
    #dUdx_dtheta[i,j] = L_i'(x / V_scale[0]) L_j( y / V_scale[0] )
    #Note  P'_{n+1} = (2n+1) P_n + P'_{n-1}
    #P'_0 = 0, P'_1 = 1
    #Note (n+1)P_{n+1} = (2n+1) x P_n  - n P_{n-1}
    Leg = np.zeros( (k_max + 1, k_max + 1) )
    Leg[0,0] = 1.0
    Leg[1,0] = x/ V_scale[0]
    for i in range(1,k_max):
        Leg[i+1,0] = ( (2*i+1)*x*Leg[i,0]/V_scale[0] - Leg[i-1,0]) / float(i+1)
    Leg[:,1] = Leg[:,0]*y / V_scale[1]
    for j in range(1,k_max):
        Leg[:,j+1] = ( (2*j+1)*y*Leg[:,j]/V_scale[1] - j*Leg[:,j-1] ) / float(j+1)

    dUdx_dtheta = np.zeros( (k_max+1, k_max+1) )
    dUdx_dtheta[1,:] = Leg[0,:]/V_scale[0]
    for i in range(1,k_max):
        dUdx_dtheta[i+1,:] = (2*i+1)*Leg[i,:]/V_scale[0] + dUdx_dtheta[i-1,:]

    dUdy_dtheta = np.zeros( (k_max+1, k_max+1) )
    dUdy_dtheta[:,1] = Leg[:,0]/V_scale[1]
    for j in range(1,k_max):
        dUdy_dtheta[:,j+1] = (2*j+1)*Leg[:,j]/V_scale[1] + dUdy_dtheta[:,j-1]
    
    return dUdx_dtheta, dUdy_dtheta

def Dtheta_predict( theta, s, x0, y0, t_arr ):
    """ Returns the sensitivity of a predicted trajectory to perturbations in theta

    args:
        theta: potential function parameters ( 2D numpy.ndarray)
        s: time scale (float)
        x0: initial x coordinate (float)
        y0: initial y coordinate (float)
        t_arr: times nodes ( 1D numpy.ndarray )

    returns:
        x_pred: x coordinates at times in t_arr (numpy.ndarray)
        y_pred: y coordinates at times in t_arr (numpy.ndarray)
        Ds_x_pred: sensitivities of x coordinates at times in t_arr (numpy.ndarray)
        Ds_y_pred: sensitivities of y coordinates at times in t_arr (numpy.ndarray)
    """
    global k_max
    def ode_func( state, t, theta, s ):
        x = state[0]
        y = state[1]
        dx_dtheta = state[2:(k_max+1)**2+2].reshape( (k_max+1, k_max+1) )
        dy_dtheta = state[(k_max+1)**2+2:].reshape( (k_max+1, k_max+1) )
        epsilon = 0.1
        dUdx = legval2d( x/V_scale[0], y/V_scale[1], legder(theta, axis=0) ) / V_scale[0]
        dUdy = legval2d( x/V_scale[0], y/V_scale[1], legder(theta, axis=1) ) / V_scale[1]
        dUdx_dtheta, dUdy_dtheta = Dtheta_grad_U( x, y )
        mag_dU = np.sqrt( dUdx**2 + dUdy**2 )
        dmag_dU_dtheta = ( dUdx*dUdx_dtheta + dUdy*dUdy_dtheta ) / (mag_dU**2)
        epsilon = 0.1
        x_dot = - s*dUdx / (mag_dU + epsilon)
        y_dot = - s*dUdy / (mag_dU + epsilon)
        dx_dot = - s*dUdx_dtheta / (mag_dU + epsilon)
        dx_dot += s*dUdx / ( (mag_dU + epsilon)**2 ) * (dUdx_dtheta*dUdx + dUdy_dtheta*dUdy) / mag_dU
        dy_dot = - s*dUdy_dtheta / (mag_dU + epsilon)
        dy_dot += s*dUdy / ( (mag_dU + epsilon)**2 ) * (dUdx_dtheta*dUdx + dUdy_dtheta*dUdy) / mag_dU
        return np.hstack( [x_dot, y_dot, dx_dot.flatten(), dy_dot.flatten() ] )
    from scipy.integrate import odeint
    state0 = np.zeros( 2*(k_max+1)**2 + 2 )
    state0[0] = x0
    state0[1] = y0
    state_arr = odeint( ode_func, state0, t_arr, args=(theta,s), rtol=1e-10 ) 
    x_pred = state_arr[:,0]
    y_pred = state_arr[:,1]
    dx_pred = state_arr[:,2:(k_max+1)**2+2].reshape( (len(t_arr),k_max+1,k_max+1) )
    dy_pred = state_arr[:,(k_max+1)**2+2:2*(k_max+1)**2+2].reshape( (len(t_arr), k_max+1, k_max+1) )
    return x_pred, y_pred, dx_pred, dy_pred


def cost( theta, s_ls, obs_ls ):
    """ computes the L2 error of the predictions

    args:
        theta: potential coefficients (numpy.ndarray)
        s: speed (float)
        obs_ls: list of curves (list(numpy.ndarray))

    returns:
        out: sum of L2 errors (float)
    """
    L2_sqr = lambda x: np.dot( x,x)
    out = 0.0
    for s,obs in zip(s_ls, obs_ls):
        x_obs,y_obs = list( obs )
        x0 = x_obs[0]
        y0 = y_obs[0]
        t_arr = np.arange( len(x_obs) )
        x_pred, y_pred = predict( theta, s, x0, y0, t_arr)
        out += 0.5*L2_sqr( x_pred - x_obs )
        out += 0.5*L2_sqr( y_pred - y_obs )
    return out

def Ds_cost( theta, s_ls, obs_ls ):
    """ computes sensitivity of L2 error of the predictions with respect to s

    args:
        theta: potential coefficients (numpy.ndarray)
        s: speed (float)
        obs_ls: list of curves (list(numpy.ndarray))

    returns:
        out: derivative of cost with respect to s (float)
    """
    out = np.zeros( len(s_ls) )
    for k,(s,obs) in enumerate(zip(s_ls,obs_ls)):
        x_obs,y_obs = list( obs )
        x0 = x_obs[0]
        y0 = y_obs[0]
        t_arr = np.arange( len(x_obs) )
        x_pred, y_pred, dx_pred, dy_pred = Ds_predict( theta, s, x0, y0, t_arr)
        out[k] = np.dot(x_pred-x_obs, dx_pred)
        out[k] += np.dot(y_pred-y_obs, dy_pred)
    return out

def Dtheta_cost( theta, s_ls, obs_ls ):
    """ computes sensitivity of L2 error of the predictions with respect to theta

    args:
        theta: potential coefficients (numpy.ndarray)
        s: speed (float)
        obs_ls: list of curves (list(numpy.ndarray))

    returns:
        out: derivative of cost with respect to theta (numpy.ndarray)
    """
    out = np.zeros_like(theta)
    for s,obs in zip(s_ls, obs_ls):
        x_obs,y_obs = list( obs )
        x0 = x_obs[0]
        y0 = y_obs[0]
        t_arr = np.arange( len(x_obs) )
        x_pred, y_pred, dx_pred, dy_pred = Dtheta_predict( theta, s, x0, y0, t_arr)
        out += np.einsum('t,tij', x_pred-x_obs, dx_pred)
        out += np.einsum('t,tij', y_pred-y_obs, dy_pred)
    return out


def regularization( theta ):
    global k_max
    k_span = np.arange(k_max+1)
    return np.einsum( 'ij,i,j', theta , k_span**2+1, k_span**2 + 1 )

def Dtheta_regularization( theta ):
    global k_max
    k_span = np.arange(k_max+1)
    return np.einsum('i,j->ij', k_span**2+1, k_span**2 + 1 )


def encode( theta,s_ls ):
    """
    encodes a decision vector for input into an optimizer

    Args:
        theta (numpy.ndarray): coefficients for potential function
        s (numpy.ndarray): time-scale parameters

    Returns:
        decision_vector (numpy.ndarray):
    """
    return np.hstack( [ theta.flatten(), s_ls] )


def decode( decision_vector ):
    """
    Inverse of encode.

    Args:
        decision_vector (numpy.ndarray)

    Returns:
        theta (numpy.ndarray): coefficients for potential function
        s_ls (numpy.ndarray): time-scale parameters
    """
    global k_max
    theta = decision_vector[:(k_max+1)**2].reshape( (k_max+1,k_max+1) )
    s_ls = decision_vector[(k_max+1)**2:]
    return theta, s_ls


#TODO: Code the following routines
lambda_regularization = 1e-5
def cost_function_for_minimizer( decision_vector, obs_ls ):
    """
    A cost function for input into scipy.optimize.minimize

    Args:
        decision_vector (numpy.ndarray):
        obs_ls (list(numpy.ndarray) ): A list of 2D curves

    Returns:
        total cost (float):
    """
    theta,s_ls = decode( decision_vector)
    return cost( theta, s_ls, obs_ls ) + lambda_regularization * regularization(theta)


def jac_cost_function_for_minimizer( decision_vector, obs_ls ):
    """
    Return the Jacobian of the cost_function_for_minizer(...)

    Args:
        decision_vector (numpy.ndarray)
        obs_ls (list(numpy.ndarray) ): A list of 2D curves

    Returns:
        jac_cost (numpy.ndarray)
    """
    theta, s_ls = decode( decision_vector )
    theta_sensitivity = Dtheta_cost( theta, s_ls, obs_ls )
    theta_sensitivity += lambda_regularization * Dtheta_regularization(theta)
    s_sensitivity = Ds_cost( theta, s_ls, obs_ls )
    return np.hstack( [ theta_sensitivity.flatten() , s_sensitivity] )


if __name__ == "__main__":
    k_max = 4
    np.random.RandomState(seed=42)
    k_span = np.arange(0,k_max+1)
    theta = np.random.randn( k_max+1, k_max+1 ) / np.outer( k_span**2+1, k_span**2+1)
    s = 0.5
    x0 = 1.5
    y0 = 1.2
    V_scale = (300, 500)
    t_arr = np.arange(0,100)

    print "TESTING prediction sensitivity to s:"
    pert = 1e-6
    x_pred_pert, y_pred_pert = predict( theta, s + pert, x0, y0, t_arr )
    x_pred, y_pred = predict( theta, s, x0, y0, t_arr)
    x_pred_comp, y_pred_comp, dx_pred, dy_pred = Ds_predict( theta, s, x0, y0, t_arr )
    fd = x_pred_pert - x_pred
    computed = dx_pred*pert
    print " finite difference = %g" % fd[-1]
    print " computed          = %g" % computed[-1]
    print " error             = %g\n" % np.abs(fd[-1]-computed[-1]).max()

    print "TESTING prediction sensitivity to theta:"
    pert = np.ones( (k_max+1,k_max+1) )*1e-5
    pert[2,2] = 0.2
    x_pred_pert, y_pred_pert = predict( theta + pert, s, x0, y0, t_arr )
    x_pred, y_pred = predict( theta, s, x0, y0, t_arr)
    x_pred_comp, y_pred_comp, dx_pred, dy_pred = Dtheta_predict( theta, s, x0, y0, t_arr )
    fd = x_pred_pert - x_pred
    computed = np.einsum( 'tij,ij', dx_pred, pert)
    print " finite difference = %g" % fd[-1]
    print " computed          = %g" % computed[-1]
    print " error             = %g\n" % np.abs(fd[-1]-computed[-1]).max()

    print "TESTING cost sensitivity to s"
    from random import randint
    obs_ls = [ np.random.randn( 2, randint(2,200) ) for _ in range(6) ]
    s_ls = np.random.rand(6)
    pert = 1e-5*np.random.randn(6)
    c1 = cost( theta, s_ls+pert, obs_ls)
    c0 = cost( theta, s_ls, obs_ls )
    fd = c1 - c0
    computed = np.dot( Ds_cost( theta, s_ls, obs_ls ), pert)
    print " finite difference = %g" % fd
    print " computed          = %g" % computed
    print " error             = %g\n" % (fd-computed)

    print "TESTING cost sensitivity to theta"
    pert = ( np.ones( (k_max+1, k_max+1)) + np.random.rand(k_max+1,k_max+1) ) * 1e-5
    pert[2,2] = 0.2
    c1 = cost( theta + pert, s_ls, obs_ls )
    c0 = cost( theta, s_ls, obs_ls )
    fd = c1 - c0
    computed = np.einsum('ij,ij', Dtheta_cost( theta, s_ls, obs_ls ), pert )
    print " finite difference = %g" % fd
    print " computed          = %g" % computed
    print " error             = %g\n" % (fd-computed)


    print "TESTING regularization sensitivity"
    R1 = regularization(theta + pert)
    R0 = regularization(theta )
    fd = R1 - R0
    computed = np.einsum( 'ij,ij', Dtheta_regularization( theta ) , pert )
    print " finite difference = %g" % fd
    print " computed          = %g" % computed
    print " error             = %g\n" % (fd-computed)

    print "TESTING cost_for_minizer_jacobian"
    decision_vector = encode( theta, s_ls )
    pert = (1.0 + np.random.rand( len(decision_vector) ) )*1e-5
    C1 = cost_function_for_minimizer( decision_vector + pert , obs_ls )
    C0 = cost_function_for_minimizer( decision_vector, obs_ls )
    fd = C1 - C0
    computed = np.dot(jac_cost_function_for_minimizer( decision_vector, obs_ls ), pert)
    print " finite difference = %g" % fd
    print " computed          = %g" % computed
    print " error             = %g" % (fd-computed)
