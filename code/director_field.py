import numpy as np

k_max = 4
V_scale = (1.0, 1.0)
regularization_coefficient = 0.1

def get_angle( points, alpha):
    """ Return angle at points given Legendre coefficients

    args:
        points (numpy.ndarray): an array of shape (2 x N_points)
        alpha (numpy.ndarray): an array of shape (k_max+1, k_max+1)

    returns:
        angle (numpy.ndarray): an array of shape (N,)
    """
    global k_max
    alpha = alpha.reshape( (k_max+1, k_max+1) )
    from numpy.polynomial.legendre import legval2d
    return legval2d( points[0] / V_scale[0] , points[1] / V_scale[1], alpha ) 

def jac_get_angle( points, alpha ):
    """ Return sensitivity of angle at points with respec to Legendre coefficients

    args:
        points (numpy.ndarray): an array of shape (2 x N_points)
        alpha (numpy.ndarray): an array of shape (k_max+1, k_max+1)

    returns:
        angle (numpy.ndarray): an array of shape (N_points, k_max+1, k_max+1)
    """
    global k_max, V_scale
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

def cost( alpha, points, directions ):
    """ Returns the cost of a director field

    args:
        alpha (numpy.ndarray): size= (k_max+1)**2
        points (numpy.ndarray): shape=(2, N_points)
        directions (numpy.ndarray): shape=(2,N_points)

    return:
        total_cost (float)
    """
    theta = get_angle( points , alpha )
    out = -( directions[0]*np.cos( theta)+directions[1]*np.sin(theta) ).sum() 
    out += regularization_coefficient*regularization( alpha )
    return out

def jac_cost( alpha, points, directions ):
    """ Returns the sensitivity of cost wrt alpha

    args:
        alpha (numpy.ndarray): size= (k_max+1)**2
        points (numpy.ndarray): shape=(2, N_points)
        directions (numpy.ndarray): shape=(2,N_points)

    return:
        out (numpy.ndarray): size=(k_max+1)**2
    """
    theta = get_angle( points, alpha )
    jac_theta = jac_get_angle( points, alpha )
    out = np.einsum('k,kij', directions[0]*np.sin(theta)-directions[1]*np.cos(theta) , jac_theta).flatten()
    out += regularization_coefficient * jac_regularization( alpha ).flatten()
    return out


def regularization( alpha ):
    """ Returns a regularization term to penalize coefficients of alpha

    args:
        alpha (numpy.ndarray): size=(k_max+1)**2

    return:
        out (float)
    """
    global k_max
    k_span = np.arange( k_max + 1 )
    alpha = alpha.reshape( (k_max+1, k_max+1) )
    return np.einsum( 'ij,i,j', 0.5*alpha**2, k_span**2 +1, k_span**2 +1 )


def jac_regularization( alpha ):
    """ Returns the jacobian of the regularization term 

    args:
        alpha (numpy.ndarray): size=(k_max+1)**2

    return:
        out (numpy.ndarray): out.shape == alpha.shape
    """
    global k_max
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
    u = trajectory[0,step::step] - trajectory[0,:n-step:step]
    v = trajectory[1,step::step] - trajectory[1,:n-step:step]
    speed = np.sqrt(u**2 + v**2)
    x = 0.5*(trajectory[0,step::step] + trajectory[0,:n-step:step])
    y = 0.5*(trajectory[1,step::step] + trajectory[1,:n-step:step])
    return np.vstack([x,y]), np.vstack([u,v])/speed 


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
