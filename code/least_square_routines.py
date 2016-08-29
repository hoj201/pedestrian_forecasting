import numpy as np
from scipy import sparse
from numpy.polynomial.legendre import legval2d, legder

def potential( x, y, theta , x_der = 0 , y_der = 0 ):
    #Computes the potential
    global V_scale
    if(x_der == 0 and y_der == 0):
        return legval2d( x / V_scale[0], y / V_scale[1] , theta)
    theta_der = legder( legder( theta , axis = 0, m=x_der), axis=1, m=y_der)
    return legval2d( x / V_scale[0] , y / V_scale[1], theta_der ) / ( V_scale[0]**x_der * V_scale[1]**y_der )


def EL_op( x_arr, y_arr, mass , theta, x0, y0, xf, yf ):
    #Computes the EL operator along a single trajectory
    assert(len(x_arr)==len(y_arr))
    x = np.hstack( [x0, x_arr, xf] )
    y = np.hstack( [y0, y_arr, yf] )
    n = len(x)
    ax = x[2:] - 2*x[1:n-1] + x[0:n-2]
    ay = y[2:] - 2*y[1:n-1] + y[0:n-2]
    DxV = potential( x[1:n-1], y[1:n-1], theta, x_der = 1 )
    DyV = potential( x[1:n-1], y[1:n-1], theta, y_der = 1 )
    return np.hstack( [mass*ax+DxV, mass*ay+DyV] )


def Dx_EL_op( x_arr, y_arr, mass, theta):
    #Computes the derivative of EL_op with respect to x_arr
    n = len(x_arr)
    diags = np.ones( (3,n) )
    diags[1] = -2*np.ones(n)
    A = sparse.dia_matrix( (diags, [-1,0,1] ), shape = (n,n) )
    V_xx = sparse.dia_matrix( ( potential( x_arr, y_arr, theta, x_der=2 ), 0 ) , shape = (n,n) ) 
    V_xy = sparse.dia_matrix( ( potential( x_arr, y_arr, theta, x_der=1, y_der=1 ), 0 ) , shape = (n,n) ) 
    return sparse.vstack( [mass*A+V_xx, V_xy] )


def Dy_EL_op( x_arr, y_arr, mass, theta):
    #Computes the derivative of EL_op with respect to y_arr
    n = len(x_arr)
    diags = np.ones( (3,n) )
    diags[1] = -2*np.ones(n)
    A = sparse.dia_matrix( (diags, [-1,0,1] ), shape = (n,n) )
    V_yy = sparse.dia_matrix( ( potential( x_arr, y_arr, theta, y_der=2 ), 0 ) , shape = (n,n) ) 
    V_xy = sparse.dia_matrix( ( potential( x_arr, y_arr, theta, x_der=1, y_der=1 ), 0 ) , shape = (n,n) ) 
    return sparse.vstack( [V_xy, mass*A+V_yy] )


def Dmass_EL_op( x_arr, y_arr, mass, theta):
    #Computes the derivative of EL_op with respect to mass
    x = np.hstack( [x0, x_arr, xf] )
    y = np.hstack( [y0, y_arr, yf] )
    n = len(x)
    ax = x[2:] - 2*x[1:n-1] + x[0:n-2]
    ay = y[2:] - 2*y[1:n-1] + y[0:n-2]
    return np.hstack( [ax ,ay ] )


def Dtheta_EL_op( x_arr, y_arr, mass, theta):
    #Computes the derivative of EL_op with respect to theta
    out_x = np.zeros( (len(x_arr), theta.shape[0], theta.shape[1] ) )
    out_y = np.zeros( (len(x_arr), theta.shape[0], theta.shape[1] ) )
    for i in range( theta.shape[0] ):
        for j in range( theta.shape[1] ):
            theta_ij = np.zeros( theta.shape )
            theta_ij[i,j] = 1
            theta_ij_x = legder( theta_ij , axis = 0 )
            theta_ij_y = legder( theta_ij , axis = 1 )
            out_x[:,i,j] = legval2d( x_arr/ V_scale[0], y_arr/ V_scale[1], theta_ij_x ) / V_scale[0]
            out_y[:,i,j] = legval2d( x_arr/ V_scale[0], y_arr/ V_scale[1], theta_ij_y ) / V_scale[1]
    return np.concatenate( [out_x, out_y], axis = 0 )



def decode_decision_vars( decision_variables, x_obs_ls, y_obs_ls , k_max ):
    #Computes x_ls, y_ls, mass_ls, theta
    offset = reduce( lambda x,y: x+y , map( len , x_obs_ls ) , 0)
    x_ls = []
    y_ls = []
    n=0
    for x_obs in x_obs_ls:
        m = len( x_obs)
        x_ls.append( decision_variables[n:n+m] )
        y_ls.append( decision_variables[n+offset:n+offset+m] )
        n +=m
    n += offset
    mass_ls = decision_variables[n:n+len(x_obs_ls)]
    n += len(x_obs_ls)
    theta = decision_variables[n:].reshape( (k_max+1, k_max+1) )
    return x_ls, y_ls, mass_ls, theta



def encode_decision_vars( x_ls, y_ls, mass_ls, theta):
    #Computes decision variables from  x_ls, y_ls, mass_ls, theta
    return np.hstack( x_ls + y_ls + list(mass_ls) + list( theta.flatten() ) )



#TODO: CODE ALL THE FUNCTIONS BELOW
def EL_constraint( decision_variables, x_obs_ls, y_obs_ls, k_max ):
    #Compute the EL_operator on each curve and stacks the result
    x_ls, y_ls, mass_ls, theta = decode_decision_vars( decision_variables, x_obs_ls, y_obs_ls, k_max )
    x0_ls = [x[ 0] for x in x_ls]
    xf_ls = [x[-1] for x in x_ls]
    y0_ls = [y[ 0] for y in y_ls]
    yf_ls = [y[-1] for y in y_ls]
    return np.hstack( [ EL_op(x,y,m,theta,x0,y0,xf,yf) for (x,y,m,x0,y0,xf,yf) in zip( x_ls, y_ls, list(mass_ls), x0_ls, y0_ls, xf_ls, yf_ls ) ] )



def jac_EL_constraint( decision_variables , x_obs_ls, y_obs_ls, k_max):
    #Computes the Jacobian of the Euler-Lagrange constraint
    #STRUCTURE [ block_diagonal ( Dx_EL_ops ) , block_diagonal( Dy_EL_ops ) , block_diagonal( Dm_EL_ops) , concatenation along axis 0 of Dtheta_EL_ops ]
    x_ls, y_ls, mass_ls, theta = decode_decision_vars( decision_variables, x_obs_ls, y_obs_ls, k_max)

    Block_0 = sparse.block_diag([ Dx_EL_op(x,y,m,theta) for (x,y,m) in zip(x_ls,y_ls,mass_ls)] )
    Block_1 = sparse.block_diag([ Dy_EL_op(x,y,m,theta) for (x,y,m) in zip(x_ls,y_ls,mass_ls)] )
    Block_2 = sparse.block_diag([ Dmass_EL_op(x,y,m,theta) for (x,y,m) in zip(x_ls,y_ls,mass_ls)] )
    Block_2 = Block_2.transpose()
    Block_3 = np.concatenate([Dtheta_EL_op(x,y,m,theta).reshape((2*len(x),theta.size)) for (x,y,m) in zip(x_ls,y_ls,mass_ls)], axis=0 )
    return np.hstack([Block_0.todense(), Block_1.todense(), Block_2.todense(), Block_3 ])


#TESTS
k_max = 3
theta = np.zeros( (k_max+1, k_max + 1) )
theta[0,1] = 1.0
V_scale = (200,300)
x_arr = np.linspace( 0, 100, 200) + 10*np.sin( np.linspace(-np.pi, np.pi,200) )
y_arr = np.linspace( 0, 50, 200) + 10*np.sin( np.linspace(-np.pi, np.pi, 200) )
mass = 1.5

x0 = x_arr[0]
xf = x_arr[-1]
y0 = y_arr[0]
yf = y_arr[-1]

print "TESTING Dx_EL_op:"
pert = np.random.randn( len(x_arr) )*1e-4
fd = EL_op( x_arr+pert, y_arr, mass, theta, x0, y0, xf, yf) - EL_op( x_arr, y_arr, mass, theta, x0, y0, xf, yf )
computed = Dx_EL_op( x_arr, y_arr, mass, theta).dot( pert )
print "\t Finite difference = %g" % fd[3]
print "\t Computed          = %g" % computed[3]
print "\t Error             = %g\n" % np.max( np.abs(fd - computed) )


print "TESTING Dy_EL_op:"
pert = np.random.randn( len(y_arr) )*1e-4
fd = EL_op( x_arr, y_arr + pert , mass, theta, x0, y0, xf, yf) - EL_op( x_arr, y_arr, mass, theta, x0, y0, xf, yf )
computed = Dy_EL_op( x_arr, y_arr, mass, theta).dot( pert )
print "\t Finite difference = %g" % fd[len(x_arr)+3]
print "\t Computed          = %g" % computed[len(x_arr)+3]
print "\t Error             = %g\n" % np.max( np.abs(fd - computed) )



print "TESTING Dmass_EL_op:"
pert = np.random.randn()*1e-4
fd = EL_op( x_arr, y_arr, mass+pert, theta, x0, y0, xf, yf) - EL_op( x_arr, y_arr, mass, theta, x0, y0, xf, yf )
computed = Dmass_EL_op( x_arr, y_arr, mass, theta).dot( pert )
print "\t Finite difference = %g" % fd[len(x_arr)+3]
print "\t Computed          = %g" % computed[len(x_arr)+3]
print "\t Error             = %g\n" % np.max( np.abs(fd - computed) )


print "TESTING Dtheta_EL_op:"
pert = np.random.randn(k_max+1,k_max+1)*1e-2
fd = EL_op( x_arr, y_arr, mass, theta+pert, x0, y0, xf, yf) - EL_op( x_arr, y_arr, mass, theta, x0, y0, xf, yf )
computed = np.einsum( 'ijk,jk', Dtheta_EL_op( x_arr, y_arr, mass, theta) , pert )
print "\t Finite difference = %g" % fd[len(x_arr)+3]
print "\t Computed          = %g" % computed[len(x_arr)+3]
print "\t Error             = %g\n" % np.max( np.abs(fd - computed) )



print "TESTING encode / decode routines:"
from random import randint
x_ls = [ np.random.randn(randint(1,100)) for _ in range(10) ]
y_ls = [ np.random.randn(len(x)) for x in x_ls ]
x_obs_ls = [ np.random.randn(len(x)) for x in x_ls ]
y_obs_ls = [ np.random.randn(len(x)) for x in x_ls ]
mass_ls = np.random.rand( len(x_ls) )

decision_vars = encode_decision_vars( x_ls, y_ls, mass_ls, theta )
x_ls_out, y_ls_out, mass_ls_out, theta_out = decode_decision_vars( decision_vars, x_obs_ls, y_obs_ls, k_max )

x_res = all( map( np.allclose , x_ls_out,  x_ls ) )
y_res = all( map( np.allclose , y_ls_out,  y_ls ) )
mass_res = np.allclose( mass_ls_out,  mass_ls )
theta_res = np.allclose( theta_out, theta )

if x_res and y_res and mass_res and theta_res:
    print "\t PASSED!\n"
else:
    print "\t FAIL!\n"


print "TESTING jacobian of EL_constraint"
pert = np.random.randn( len(decision_vars) )*1e-6
out1 = EL_constraint( decision_vars+pert , x_obs_ls, y_obs_ls, k_max )
out0 = EL_constraint( decision_vars-pert , x_obs_ls, y_obs_ls, k_max )
fd = 0.5*(out1-out0)
Jacobian = jac_EL_constraint( decision_vars, x_obs_ls, y_obs_ls, k_max)
computed =  np.dot(Jacobian,pert)

print "\t Finite difference = %g" % fd[3]
print "\t Computed          = %g" % computed[0,3]
print "\t Error             = %g\n" % np.max( np.abs(fd - computed) )

print np.mean( np.abs(fd - computed) )
