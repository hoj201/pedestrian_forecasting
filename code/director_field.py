import numpy as np
from numpy.polynomial.legendre import legval2d, legder

regularization_coefficient = 1.0

def get_angle( points, alpha, k_max, V_scale):
    """ Return angle at points given Legendre coefficients

    args:
        points (numpy.ndarray): an array of shape (2 x N_points)
        alpha (numpy.ndarray): an array of shape (k_max+1, k_max+1)

    returns:
        angle (numpy.ndarray): an array of shape (N,)
    """
    alpha = alpha.reshape( (k_max+1, k_max+1) )
    from numpy.polynomial.legendre import legval2d
    return legval2d( points[0] / V_scale[0] , points[1] / V_scale[1], alpha ) 

def jac_get_angle( points, alpha, k_max, V_scale ):
    """ Return sensitivity of angle at points with respec to Legendre coefficients

    args:
        points (numpy.ndarray): an array of shape (2 x N_points)
        alpha (numpy.ndarray): an array of shape (k_max+1, k_max+1)
        V_scale (tuple<int>) : describes the size of the domain

    returns:
        angle (numpy.ndarray): an array of shape (N_points, k_max+1, k_max+1)
    """
    N_points = points.shape[1]
    alpha = alpha.reshape( (k_max+1, k_max+1) )
    x = points[0]
    y = points[1]
    Leg = np.ones( (N_points, k_max + 1, k_max + 1) )
    Leg[:,1,0] = x/ V_scale[0]
    for i in range(1,k_max):
        Leg[:,i+1,0] = ( (2*i+1)*x*Leg[:,i,0]/V_scale[0] -i*Leg[:,i-1,0]) / float(i+1)
    Leg[:,:,1] = np.einsum('pj,p->pj', Leg[:,:,0], y/ V_scale[1] )
    for j in range(1,k_max):
        Leg[:,:,j+1] = ( (2*j+1)*np.einsum('pj,p->pj',Leg[:,:,j], y/V_scale[1]) - j*Leg[:,:,j-1] ) / float(j+1)
    return Leg

def cost( alpha, points, directions, k_max, V_scale ):
    """ Returns the cost of a director field

    args:
        alpha (numpy.ndarray): size= (k_max+1)**2
        points (numpy.ndarray): shape=(2, N_points)
        directions (numpy.ndarray): shape=(2,N_points)
        k_max (int): max degree of polynomial for angle-field
        V_scale (tuple<int>) : describes the size of the domain

    return:
        total_cost (float)
    """
    theta = get_angle( points , alpha, k_max, V_scale )
    out = -( directions[0]*np.cos( theta)+directions[1]*np.sin(theta) ).sum() 
    out += regularization_coefficient*regularization( alpha, k_max )
    return out

def jac_cost( alpha, points, directions, k_max, V_scale ):
    """ Returns the sensitivity of cost wrt alpha

    args:
        alpha (numpy.ndarray): size= (k_max+1)**2
        points (numpy.ndarray): shape=(2, N_points)
        directions (numpy.ndarray): shape=(2,N_points)
        k_max (int): max degree of polynomial for angle-field
        V_scale (tuple<int>) : describes the size of the domain

    return:
        out (numpy.ndarray): size=(k_max+1)**2
    """
    theta = get_angle( points, alpha, k_max, V_scale )
    jac_theta = jac_get_angle( points, alpha, k_max, V_scale )
    out = np.einsum('k,kij', directions[0]*np.sin(theta)-directions[1]*np.cos(theta) , jac_theta).flatten()
    out += regularization_coefficient * jac_regularization( alpha, k_max ).flatten()
    return out


def regularization( alpha, k_max ):
    """ Returns a regularization term to penalize coefficients of alpha

    args:
        alpha (numpy.ndarray): size=(k_max+1)**2
        k_max : int

    return:
        out (float)
    """
    k_span = np.arange( k_max + 1 )
    alpha = alpha.reshape( (k_max+1, k_max+1) )
    return np.einsum( 'ij,i,j', 0.5*alpha**2, k_span**2 +1, k_span**2 +1 )


def jac_regularization( alpha, k_max ):
    """ Returns the jacobian of the regularization term 

    args:
        alpha (numpy.ndarray): size=(k_max+1)**2
        k_max: (int)

    return:
        out (numpy.ndarray): out.shape == alpha.shape
    """
    k_span = np.arange( k_max + 1)
    alpha = alpha.reshape( (k_max+1, k_max+1) )
    return np.einsum('ij,i,j->ij', alpha, k_span**2+1, k_span**2+1).reshape( alpha.shape )


def trajectory_to_directors( trajectory, step = 5 ):
    """ returns directors along a given trajectory

    args:
        trajectory (numpy.ndarray) : shape = (2,n)

    kwargs:
        step (int): step size for finite differencing

    return:
        u (numpy.ndarray): x-component of direction
        v (numpy.ndarray): y-component of direction
    """
    n = trajectory.shape[1]
    x = 0.5*(trajectory[0,step::step] + trajectory[0,:n-step:step])
    y = 0.5*(trajectory[1,step::step] + trajectory[1,:n-step:step])
    u = trajectory[0,step::step] - trajectory[0,:n-step:step]
    v = trajectory[1,step::step] - trajectory[1,:n-step:step]
    speed = np.sqrt(u**2 + v**2) 
    remove_baddies = lambda x: x[np.nonzero(speed)]
    x = remove_baddies(x)
    y = remove_baddies(y)
    u = remove_baddies(u)
    v = remove_baddies(v)
    speed = remove_baddies(speed)
    return np.vstack([x,y]), np.vstack([u,v])/speed 

def ode_function( xy, t, alpha, speed ):
    """ returns velocity feild for input into odeint

    args:
        xy (np.ndarray) : shape = (2,)
        t (float):
        alpha: coefficients for angles

    returns:
        out (np.ndarray) : velocity
    """
    global V_scale
    x = xy[0]
    y = xy[1]
    theta = legval2d( x / V_scale[0], y / V_scale[1], alpha )
    out = speed*np.array( [np.cos(theta), np.sin(theta) ] )
    return out


def jac_ode_function( xy, t, alpha, speed ):
    """ returns velocity feild for input into odeint

    args:
        xy (np.ndarray) : shape = (2,)
        t (float):
        alpha: coefficients for angles

    returns:
        out (np.ndarray) : jacobian of velocity field, shape = (2,2)
    """
    global V_scale
    x = xy[0]
    y = xy[1]
    theta = legval2d( x / V_scale[0], y / V_scale[1], alpha )
    theta_x = legval2d( x / V_scale[0], y / V_scale[1], legder( alpha, axis=0) ) / V_scale[0]
    theta_y = legval2d( x / V_scale[0], y / V_scale[1], legder( alpha, axis=1) ) / V_scale[1]
    out = np.zeros( (2,2) )
    out[0,0] = - np.sin(theta)*theta_x
    out[0,1] = - np.sin(theta)*theta_y
    out[1,0] = np.cos(theta)*theta_x
    out[1,1] = np.cos(theta)*theta_y
    out *= speed
    return out

def rk4_predict( x0, y0, alpha, speed ):
    global V_scale
    def rk4_step( xy , h ):
        k1 = ode_function( xy, 0.0, alpha, speed)
        k2 = ode_function( xy + h*k1/2, 0.0, alpha, speed)
        k3 = ode_function( xy + h*k2/2, 0.0, alpha, speed)
        k4 = ode_function( xy + h*k3, 0.0, alpha, speed)
        return xy + h*(k1+2*k2+2*k3+k4)/6.0

    def in_view( xy):
        x,y = xy[0],xy[1]
        if np.abs(x) < V_scale[0] and np.abs(y) < V_scale[1]:
            return True
        return False

    xy_arr = [ np.array([x0,y0]) ]
    while in_view( xy_arr[-1] ):
        xy_arr.append( rk4_step( xy_arr[-1], 0.5 ) )
    xy_arr = np.array( xy_arr )
    return xy_arr[:,0], xy_arr[:,1]


def trajectories_to_director_field( trajectories, V_scale, step = 10, k_max = 6 ):
    """ Converts a collection of trajectories into a director field

    args:
        trajectories (list of numpy.ndarray): shape of each element is (2,N_time_nodes)

    kwargs:
        step (int) : step size for approximation of tangents

    returns:
        alpha (numpy.ndarray) : coeficients for 2D legendre series, shape = (k_max+1, k_max+1)
    """
    points_ls, directions_ls = zip(*[trajectory_to_directors(traj,step=step) for traj in trajectories ] )
    points = np.hstack(points_ls)
    directions = np.hstack( directions_ls)
    alpha_guess = np.zeros( (k_max+1, k_max+1) )
    av_dir = np.power( reduce( lambda x,y: x*y, directions[0]+1j*directions[1]), 1.0 / directions.shape[1])
    alpha_guess[0,0] = np.log( av_dir ).imag
    from scipy.optimize import minimize
    res = minimize( cost, alpha_guess, jac = jac_cost, args = (points, directions,k_max,V_scale), method ='Newton-CG')
    if not res.success:
        print res.message
    alpha = res.x.reshape( (k_max+1, k_max+1) )
    return alpha


def polyfit2d(x, y, f, deg):
    """ Returns polynomial coefficients given the values of a function at some point

    Code courtesy of 'klaus se' on stackoverflow.com/questions/7997152

    args:
        x (numpy.ndarray) : x coordinates
        y (numpy.ndarray) : y coordinates
        f (numpy.ndarray) : funtion values 
        deg (iterable of ints) : (max_degree_x, max_degre_y)

    returns:
        c (numpy.ndarray) : array of polynomial coefficients for use with numpy's polyval2d routine
    """
    from numpy.polynomial import polynomial
    import numpy as np
    x = np.asarray(x)
    y = np.asarray(y)
    f = np.asarray(f)
    deg = np.asarray(deg)
    vander = polynomial.polyvander2d(x, y, deg)
    vander = vander.reshape((-1,vander.shape[-1]))
    f = f.reshape((vander.shape[0],))
    c = np.linalg.lstsq(vander, f)[0]
    return c.reshape(deg+1)


import hermite_function
def director_field_to_FP_operator( alpha, deg = [50,50], poly_deg = [5,5] ):
    """ Exports a Fokker-Planck operator

    args:
        alpha (numpy.ndarray) : Legendre coeffcients for the angles of the director field

    kwargs:
        deg (list(int)) : resolution of output
        poly_deg (list(int)) : degree of polynomial approximation of vector-field

    returns:
        FP_op (hermite.FP_operator)

     NOTES:
        The hermite_funtion routines do not work well when the domain is large.
        Therefore we re-scale the vector-field down to the [-1 , 1] x [-1, 1] square.
    """
    global V_scale
    M = [1.0, 1.0]
    X_grid, Y_grid = np.meshgrid( np.linspace( -M[0], M[0], 50 ),
            np.linspace( -M[0], M[1], 50) )
    points = np.vstack( [ V_scale[0]*X_grid.flatten() / M[0], V_scale[1]*Y_grid.flatten() / M[1] ] )
    theta = get_angle( points, alpha )
    X_poly = polyfit2d( X_grid, Y_grid, np.cos(theta), poly_deg) / V_scale[0] 
    Y_poly = polyfit2d( X_grid, Y_grid, np.sin(theta), poly_deg) / V_scale[1]
    FP_op = hermite_function.FP_operator( polynomials = [X_poly, Y_poly], M=M, deg = deg )
    return FP_op


if __name__ == "__main__":
    t_span = np.linspace( 0 , np.pi , 40)
    trajectory = np.vstack( [ np.cos( t_span ) , np.sin(t_span ) ] )
    points, directions = trajectory_to_directors( trajectory, step=2 )

    print "Testing jac_get_angle"
    k_span = np.arange(k_max+1)
    alpha = np.zeros((k_max+1,k_max+1))
    pert = 1e-6*np.random.rand( *alpha.shape )
    theta1 = get_angle( points, alpha + 0.5*pert )
    theta0 = get_angle( points, alpha - 0.5*pert )
    fd = theta1 - theta0
    computed = np.einsum('iab,ab', jac_get_angle( points, alpha ) , pert )

    print "  finite difference = %g" % fd[3]
    print "  computed          = %g" % computed[3]
    print "  error             = %g" % np.abs(fd[3] - computed[3]) 

    print "Testing jac cost"
    C1 = cost( alpha + 0.5*pert, points, directions )
    C0 = cost( alpha - 0.5*pert, points, directions )
    fd = C1 - C0
    computed = np.dot( jac_cost( alpha.flatten() , points, directions ), pert.flatten() )
    print "  finite difference = %g" % fd
    print "  computed          = %g" % computed
    print "  error             = %g" % np.abs(fd - computed) 


    print "Testing director_field_to_FP_operator( alpha )"
    alpha = np.zeros( (k_max+1,k_max+1) )
    FP_op = director_field_to_FP_operator( alpha ) #this should be the vector-field ddx
    #Lay down a Gaussian at the origin
    sigma = 0.5
    mu_x = 0.0
    mu_y = 0.0
    f = lambda x,y: np.exp( -((x-mu_x)**2 + (y-mu_y)**2 ) / (2*sigma**2) )
    h_series = hermite_function.hermite_function_series( M=[1.0, 1.0], deg=[50, 50] )
    h_series.interpolate( f )

    #Advect the gaussian for time 1 and plot begin and end pdfs
    from scipy.integrate import odeint
    x0 = h_series.coeffs.flatten()
    ode_func = lambda x,t : FP_op.op.dot(x)
    t = np.linspace(0,0.1,10)
    x = odeint( ode_func, x0, t )

    from matplotlib import pyplot as plt
    X_grid, Y_grid = np.meshgrid( np.linspace(-1,1,50), np.linspace(-1,1,50) )
    xf = x[-1]
    fig,ax = plt.subplots( 2, figsize = (10,5) )
    Z_grid0 = h_series.evaluate_on_grid( [X_grid, Y_grid] )
    shape = h_series.coeffs.shape
    new_h_series = hermite_function.hermite_function_series( coeffs=xf.reshape( shape ), M=[1.0,1.0], deg=[50,50] ) 
    Z_grid0 = h_series.evaluate_on_grid( [X_grid, Y_grid] )
    Z_gridf = new_h_series.evaluate_on_grid( [X_grid, Y_grid] )

    ax[0].imshow( Z_grid0, extent=[-1,1,-1,1], cmap = 'viridis' )
    ax[0].scatter( 0,0)
    ax[1].imshow( Z_gridf, extent=[-1,1,-1,1], cmap = 'viridis' )
    ax[1].scatter( 0.1, 0 )
    plt.show()


