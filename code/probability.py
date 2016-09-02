import numpy as np

V_scale = (1.0, 1.0 )
k_max = 4
sigma_x = V_scale[0] / 10
sigma_v = V_scale[0] / 10
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


def x_given_ck( x , alpha ):
    """ Computes the posterior probability  P( position | class )
    
    args:
        x (numpy.ndarray): array of points
        alpha (numpy.ndarray): array of coefficients for the director-fields, shape = (n, k_max+1,k_max+1)

    returns:
        out (numpy.ndarray): probability of being at position given class.  shape = (N_points,n).
    """
    global V_scale
    #TODO:  This need to able to handle a 3-dimensional alphs
    Z = partition_function( alpha )
    V = np.polynomial.legendre.legval2d( x[0] / V_scale[0], x[1] / V_scale[1], alpha)
    return np.exp( -V ) / Z

def ck_given_x( alpha, x ):
    """ Computes the probability of a class given a location of an agent
    
    args:
        x (numpy.ndarray): array of points
        alpha (numpy.ndarray): array of coefficients for angle of class. shape=(n,k_max+1, k_max+1)

    returns:
        out (float): probability of being at position given class defined by alpha, shape=(n,N_points).
    """
    #TODO: Use Baye's theorem an the x_given_ck routine
    return -1



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


def x_T_given_mu_eta_c0( x_T, mu, eta):
    """ Computes the probability of x_T given measurements under linear motion dynamics

    args:
        x_T (numpy.ndarray) : (2,N) array of points
        mu (numpy.ndarray) : (2,) array of measurment of initial position
        eta (numpy.ndarray) : (2,) array of measurement of initial velocity

    returns:
        p (numpy.ndarray) : (N,) array of floats between 0.0 and 1.0
    """
    global V_scale, sigma_x, sigma_v
    p = np.zeros( len(x_T[0]) )
    #TODO:  See formula with happy face

    return -1


def c0_given_mu_eta( mu, eta ):
    """ Compute the probability of class = c0 given measurements

    args:
        mu (numpy.ndarray) : (2,) array of measurment of initial position
        eta (numpy.ndarray) : (2,) array of measurement of initial velocity

    returns:
        p (float) : between 0.0 and 1.0
    """
    #TODO:  See formula with happy face
    return -1


def ck_given_mu_eta( alpha , mu, eta ):
    """ Computes the probabilities of classes c1,...,cn given measuremenst

    args:
        alpha (numpy.ndarray) : legendre coeffs for angle field, shape=(n,k_max+1,k_max+1)
        mu (numpy.ndarray) : (2,) array of measurment of initial position
        eta (numpy.ndarray) : (2,) array of measurement of initial velocity

    returns:
        p (numpy.ndarray) : floats between 0.0 and 1.0, size=n
    """
    #TODO: See formula with a penis by it
    return -1

def ck_and_s_given_mu_eta( alpha, s, mu, eta ):
    """ Computes the joint probabilities of classes c1,...,cn and speed=s, given measuremenst

    args:
        alpha (numpy.ndarray) : legendre coeffs for angle field, shape=(n,k_max+1,k_max+1)
        s (float) : speed
        mu (numpy.ndarray) : (2,) array of measurment of initial position
        eta (numpy.ndarray) : (2,) array of measurement of initial velocity

    returns:
        p (numpy.ndarray) : floats between 0.0 and 1.0, size=n, one for each class
    """
    #TODO: See formula with a coffee cup by it
    return -1


def s_given_c_eta_mu( s, alpha, eta, mu )
    """ Computes the joint probabilities of speed=s, for each class, given measurement

    args:
        s (float) : speed
        alpha (numpy.ndarray) : legendre coeffs for angle field, shape=(n,k_max+1,k_max+1)
        s (float) : speed
        mu (numpy.ndarray) : (2,) array of measurment of initial position
        eta (numpy.ndarray) : (2,) array of measurement of initial velocity

    returns:
        p (numpy.ndarray) : floats between 0.0 and 1.0, size=n, one for each class
    """
    #TODO: See formula with a angry face and a fish.
    return -1


def x_given_ck_mu( x, alpha, mu ):
    """ Computes the probability of position x given class c1,...,cn and measurement mu

    args:
        x (numpy.ndarray) : (2,N) array of points
        alpha (numpy.ndarray) : legendre coeffs for angle field, shape=(n,k_max+1,k_max+1)
        mu (numpy.ndarray) : (2,) array of measurment of initial position

    returns:
        p (numpy.ndarray) : floats between 0.0 and 1.0, size=n, one for each class
    """
    #TODO: See formula with flower next to it.
    return -1



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
