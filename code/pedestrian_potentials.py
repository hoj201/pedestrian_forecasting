import numpy as np

V_scale = (1.0, 1.0 )
k_max = 4
print "Remember to set V_scale and k_max"

def learn_potential( x_arr, y_arr , k_max = 6 ):
    """Returns the legendre coefficients of a potential function learned from a list of point in a 2D domain.

    args:
        x_arr (iterable): sequence of x coordinates
        y_arr (iterable): sequence of y coordinates (len(x_arr)==len(y_arr))
        width (numeric):
        height (numeric) : 
    returns:
        out (numpy.ndarray) : coeffs for 2D Legendre series describing potential funciton.
    """
    global V_scale

    # COST FUNCTION 
    def cost_function(theta_flat ):
        V_sum = 0
        N = 0
        theta = theta_flat.reshape( (k_max+1, k_max+1))
        from numpy.polynomial.legendre import legval2d,leggauss
        V_mean = legval2d( x_arr/V_scale[0], y_arr/V_scale[1], theta).mean()
        k_span = np.arange( k_max+1)
        res = 2*(k_max+10)
        x_span = np.linspace(-V_scale[0], V_scale[0], res)
        y_span = np.linspace(-V_scale[1], V_scale[1], res)
        x_grid,y_grid = np.meshgrid(x_span, y_span)
        I = width*height*np.exp( - legval2d( x_grid/V_scale[0], y_grid/V_scale[1] , theta)).sum() / (res**2)
        regularization = np.sqrt( np.einsum( 'ij,i,j', theta**2 , k_span**2 , k_span**2 ) )
        lambda_0 = 1e-4
        return V_mean + np.log(I) + lambda_0 * regularization

    # CONSTRAINTS
    def potential_constraint(theta_flat ):
        return theta_flat[0]
    constraint_list = []
    constraint_list.append({'type':'eq', 'fun':potential_constraint })
    initial_guess = np.zeros( (k_max+1)**2 )

    # CALLBACK FUNCTIONS
    def cb_function( theta_flat ):
        global k_max,curves
        cb_function.iteration += 1
        from progress_bar import update_progress
        update_progress( cb_function.iteration / float(cb_function.max_iteration) )
        return 0
    cb_function.iteration = 0
    cb_function.max_iteration = 1000

    # MINIMIZE COST
    from scipy.optimize import minimize
    res = minimize( cost_function, initial_guess, constraints=constraint_list, callback = cb_function, options={'maxiter':cb_function.max_iteration})

    # RETURN RESULT
    print res.message
    assert(res.success)
    return res.x.reshape( (k_max+1, k_max+1) )


def posterior( x , y , alpha ):
    """ Computes the posterior probability  P( position | class )
    
    args:
        x (float): x-coordinate
        y (float): y-coordinate
        alpha (numpy.ndarray): array of coefficients for the potential function of the class

    returns:
        out (float): probability of being at position (x,y) given class defined by alpha.
    """
    global V_scale
    Z = partition_function( alpha )
    V = np.polynomial.legendre.legval2d( x / V_scale[0], y / V_scale[1], alpha)
    return np.exp( -V ) / Z


def memoize(f):
    memo = {}
    def helper(x):
        if x.tostring() not in memo:
            memo[x.tostring()] = f(x)
        return memo[x.tostring()]
    return helper


@memoize
def partition_function( alpha ):
    global V_scale
    res = 50
    x_grid, y_grid = np.meshgrid( np.linspace( -V_scale[0], V_scale[0],res),
            np.linspace(-V_scale[1], V_scale[1], res ) )
    V_grid = np.polynomial.legendre.legval2d( x_grid / V_scale[0], y_grid / V_scale[0], alpha)
    return 4*V_scale[0]*V_scale[1]*np.exp(-V_grid).mean()



def prune_trajectories( curves ):
    """Given a list of curves, prune_trajectories(curves) returns a sublist of curves where the outlying curves (in terms of length) are removed

    args:
    curves -- list of numpy.ndarray

    kwargs:
    None
    """

    #Compute IQR
    log_length = lambda c: np.log( c.shape[1] )
    log_curve_lengths = map( log_length , curves )
    log_curve_lengths.sort()
    num_curves = len(curves)
    IQ_1 = log_curve_lengths[ num_curves / 4 ]
    IQ_3 = log_curve_lengths[ 3*num_curves / 4 ]
    IQR = IQ_3 - IQ_1

    #Determin which curves fall within IQR+/- 1.5*width range
    is_not_outlier = lambda c : log_length(c) > IQ_1 - 1.5*IQR or log_length(c) < IQ_3 + 1.5*IQR
    return filter( is_not_outlier , curves )


def Stormer_Verlet(x0, y0, x1, y1, n_steps, theta, V_scale, Delta_t=1.0):
    from numpy.polynomial.legendre import legder,legval2d
    theta_x = legder( theta, axis=0, m=1)
    theta_y = legder( theta, axis=1, m=1)
    x_pred = np.zeros(n_steps)
    y_pred = np.zeros(n_steps)
    x_pred[0],x_pred[1] = (x0,x1)
    y_pred[0],y_pred[1] = (y0,y1)    
    for k in range(n_steps-2):
        x1,y1 = (x_pred[k+1],y_pred[k+1])
        x0,y0 = (x_pred[k],y_pred[k])
        V_x = legval2d( x1/V_scale[0], y1/V_scale[1], theta_x )/V_scale[0]
        V_y = legval2d( x1/V_scale[0], y1/V_scale[1], theta_y )/V_scale[1]
        x_pred[k+2] = 2*x1 - x0 - Delta_t**2 * V_x
        y_pred[k+2] = 2*y1 - y0 - Delta_t**2 * V_y
    return x_pred, y_pred


if __name__ == "__main__":
    print "Testing partition function"
    alpha = np.zeros( (k_max+1, k_max+1) )
    Z = partition_function( alpha )
    print "Z = %f" % Z
    print "Correct answer is Z=4"

    print "Testing posterior(x,y, alpha)"
    x = np.random.rand()
    y = np.random.rand()
    p = posterior( x, y, alpha )
    print "p = %f" % p
    print "Correct answer = 0.25"
