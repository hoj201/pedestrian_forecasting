import numpy as np
from horners import eval_nd_poly

def hermite_function(x, deg=20):
    # computes h_n(x) for n = 0,...,deg
    out = np.zeros( (deg + 1 , x.size) )
    out[0,:] = np.pi**-0.25 * np.exp( - x**2 / 2.0)
    if deg==0:
        return out
    out[1,:] = np.sqrt(2)*x*out[0,:]
    for n in range(2,deg+1):
        out[n,:] = np.sqrt(2.0/n) * x * out[n-1,:] - np.sqrt( (n-1)/float(n)) * out[n-2,:]
    return out


def hermite_polynomial(x,deg):
    #computes the Hermite polynomials 0,...,deg at an array of points x.
    #returns a matrix of size (len(x),deg)
    H = np.zeros((deg+2, x.size) )
    H[0,:] = 1.0
    H[1,:] = 2*x
    for n in range(2,deg+2):
        H[n,:] = 2*x*H[n-1,:]-2*(n-1)*H[n-2,:]
    return H

class hermite_function_series:
    def __init__(self, coeffs = None , M=(1,), deg=(20,)):
        if isinstance(M , (int, long, float)):
            M = [M,]
        if isinstance(deg,int):
            deg = [deg,]
        self.M = list(M)
        self.deg = list(deg)
        self.dim = len(M)
        shape_tuple = map( lambda x: x+1 , deg )
        if(coeffs is None):
            self.coeffs = np.zeros( shape_tuple )
        else:
            self.coeffs = coeffs.reshape( shape_tuple )

    def set_coeffs(self , x ):
        self.coeffs = x.reshape( self.coeffs.shape )
        return 0

    def interpolate(self, f):
        #computes the hermite series which interpolates f on [-M,M]^d.
        #Output: coefficient array a[n] where f(x) = a[n] h_n(x*alpha)
        #where alpha = sqrt(2*deg)/M
        from numpy import meshgrid,sqrt
        from numpy.polynomial.hermite import hermgauss
        gamma_w = []
        for d in self.deg:
            gamma_w.append( hermgauss( d ) )
        gamma,w = zip(*gamma_w)

        h = []
        C = []
        alpha = []
        for k,d in enumerate( self.deg ):
            h.append( hermite_function( gamma[k] , d ) )
            C.append( ( h[k]**2).sum(axis=0) )
            alpha.append( sqrt( 2*d) / self.M[k] )
        f_at_gamma = f( *meshgrid( * map( lambda x:x[0]/x[1], zip(gamma,alpha) ) ) )
        h_list = []
        C_inverse_list = []
        f_index = []
        output_index = []
        for n in range( len(self.deg)):
            i = n + len(self.deg)
            h_list.append( h[n] )
            h_list.append( [n,i] )
            C_inverse_list.append( C[n]**-1 )
            C_inverse_list.append( [i] )
            f_index.append( i )
            output_index.append(n)
        arg = h_list + C_inverse_list + [f_at_gamma, f_index, output_index]
        self.coeffs = np.einsum( *arg )
        return 0

    def evaluate_on_grid( self, grid_list ):
        #given a hermite series a[m,n] we evaluate it on a grid
        #Output: array f[i] = \sum_m a[m] h_m( x_i * alpha )
        #where alpha = sqrt(2*res)/M
        out = self.coeffs
        alpha = np.sqrt( 2*self.deg[0] )/ self.M[0]
        h_nx = hermite_function( grid_list[0].flatten()*alpha , self.deg[0] )
        out = np.einsum('n...,ni->...i', out , h_nx)
        for k in range(1,len(grid_list)):
            x = grid_list[k].flatten()
            alpha = np.sqrt( 2*self.deg[k] )/ self.M[k] 
            h_nx = hermite_function( x.flatten()*alpha , self.deg[k] ) #first index is hermite index, later indices are spatial
            out = np.einsum( 'n...i,ni->...i', out, h_nx ) 
        return out.reshape( grid_list[0].shape )

    def marginalize(self, axis):
        #integrate over a single axis
        global compute_integrals_of_hermite_function
        dim = len(self.M)
        integrals = compute_integrals_of_hermite_function( self.deg[axis] )
        indices = list(range(0,dim))
        output_indices = indices + [] #deep copy of indices
        output_indices.remove(axis)
        alpha = np.sqrt( 2*self.deg[axis] )/ self.M[axis]
        new_coeffs = np.einsum(self.coeffs, indices,  integrals, [axis,], output_indices) / alpha
        new_M = [self.M[i] for i in output_indices]
        new_deg = [self.deg[i] for i in output_indices]
        return hermite_function_series( coeffs=new_coeffs, M=new_M, deg=new_deg )

    def get_uniform_grid( self, res=20):
        from numpy import meshgrid,linspace
        return meshgrid( * map( lambda m: linspace(-m,m,res) , self.M ) )

    def kron( self, other ):
        #returns the Kronecker product of two series (i.e. f(x,y) = c_i h_i(x) d_j h_j(y) )
        assert( type(other) == type(self) )
        M = self.M + other.M
        deg = self.deg + other.deg
        return hermite_function_series( coeffs = np.kron( self.coeffs , other.coeffs ), M=M, deg=deg)

    def __add__( self , other):
        assert( type(other) == type(self) )
        assert( self.M == other.M )
        assert( self.deg == other.deg )
        return hermite_function_series( coeffs = self.coeffs + other.coeffs, M = self.M, deg = self.deg)

    def __sub__( self , other):
        assert( type(other) == type(self) )
        assert( self.M == other.M )
        assert( self.deg == other.deg )
        return hermite_function_series( coeffs = self.coeffs - other.coeffs, M = self.M, deg = self.deg)

    def __mul__( self , x ):
        #scalar multiplication
        from numbers import Number
        if isinstance(x , Number):
            # if x is a scalar, then just scale the coefficients
            return hermite_function_series( coeffs = x*self.coeffs, M = self.M, deg = self.deg )

    def __lmul__(self , x ):
        return hermite_function_series( coeffs = x*self.coeffs, M = self.M, deg = self.deg )

    def __rmul__(self , x ):
        return hermite_function_series( coeffs = x*self.coeffs, M = self.M, deg = self.deg )

class FP_operator:
    #Produces a Fokker-Planck operator for densities with respect to polynomial vector fields and Gaussian noise
    def __init__(self, polynomials=None , M=(1,), deg=(20,) , sigma = None ):
        from numbers import Number
        if isinstance(M,Number):
            M = [M,]
        if isinstance(deg,int):
            deg = [deg,]
        assert( len(deg) == len(M) )
        self.deg = list(deg)
        self.M = list(M)
        dim = len(deg)
        from scipy import sparse
        n_modes = reduce( lambda x,y: x*(y+1) , deg , 1 )
        self.__number_of_states__ = n_modes
        self.op = sparse.dia_matrix( ( n_modes , n_modes ) )
        if( polynomials != None ):
            assert( len( polynomials) == len(self.deg) )
            alpha = np.sqrt(2*np.array(deg))/np.array(M)
            identity_operators = [ sparse.eye( deg[k]+1 ) for k in range( dim ) ]
            multiplication_operators = []
            for k in range( dim ):
                x = one_dimensional_mult_op( deg[k], alpha[k] )
                multiplication_operators += [ reduce( sparse.kron,
                    map( lambda arg: x if arg[0]==k else arg[1],
                        zip(range(dim), identity_operators)
                        )
                    )
                ]

            for k in range( dim ):
                mult_by_poly = eval_nd_poly( polynomials[k] , multiplication_operators , mul = lambda a,b: a.dot(b) )
                ddx = one_dimensional_diff_op( deg[k] , alpha[k] ) 
                derivative_op = reduce( sparse.kron ,
                        map( lambda arg: ddx if arg[0]==k else arg[1],
                            zip(range(dim),identity_operators)
                            )
                        )
                self.op -= derivative_op.dot( mult_by_poly )
                if(sigma is not None):
                    self.op += 0.5*sigma[k]**2 * derivative_op.dot( derivative_op )
 
    def __add__(self, other ):
        assert( type(self) == type(other) )
        assert( self.M == other.M and self.deg==other.deg)
        out = Lie_derivative( M=self.M, deg=self.deg )
        out.op = self.op + other.op
        return out

    def dot( self, h_series ):
        #applies the Lie derivative Operatot to a hermite series
        assert( self.M == h_series.M )
        assert( self.deg == h_series.deg )
        shape_tuple = h_series.coeffs.shape
        coeffs = self.op.dot( h_series.coeffs.flatten() ).reshape( shape_tuple )
        return hermite_function_series( coeffs = coeffs, M = self.M , deg=self.deg )

    def advect( self, h_series, t ):
        #evolves h_series by time t according to the Schrodinger equation
        assert( self.M == h_series.M )
        assert( self.deg == h_series.deg )
        x0 = h_series.coeffs.flatten()
        from scipy.sparse.linalg import expm_multiply
        xt = expm_multiply( self.op , x0, start=0, stop=t, num=2, endpoint=True )[1]
        return hermite_function_series( coeffs=xt.reshape( h_series.coeffs.shape ), M=self.M, deg=self.deg)

    def cayley_step(self, h_series, dt ):
        from scipy.sparse.linalg import solve
        #save cayley op as an attribute
        #Q = (I+0.5*A)*(I-0.5*A)**-1
        return h_series

def compute_hermite_coeffs(n_max):
    #computes the coefficients a[n][k] where h_n(x) = \exp(-x^2 / 2) \sum_{k=0}^{n} a_{n,k} x^{k}
    #for n=0,...,n_max
    assert( n_max >= 0)
    a_list = [ np.array( [np.pi**-0.25] ) , np.array([0.0, np.sqrt(2)*np.pi**(-0.25)]), ]
    for n in range(1,n_max+1):
        #shift = np.diag( np.ones(n+1)  , k=-1)
        a_n = a_list[-1]
        a_n_minus_1 = np.hstack( [a_list[-2], np.zeros(2)] )
        a_n_plus_1 = np.sqrt(2/float(n+1)) * np.hstack( [ np.zeros(1) ,a_n] ) - np.sqrt(n/float(n+1)) * a_n_minus_1
        a_list.append( a_n_plus_1 )
    return a_list

def compute_integrals_of_hermite_function( n_max):
    #computes the integrals \int h_n(x) for n=0,...,n_max
    a_list = compute_hermite_coeffs(n_max)
    out = np.zeros(n_max+1)
    b_0 = np.sqrt(np.pi*2)
    for n in range(0,n_max+1,2):
        g = a_list[n][n]
        for k in range(2,n+1,2):
            g = (n-k+1)*g+a_list[n][n-k]
        out[n] = g*b_0
    return out

def one_dimensional_diff_op( degree, alpha ):
    #produces the one dimensional differential operator in the basis h_n( alpha* x) for n=0,...,degree
    from scipy.sparse import diags
    down_shift = diags( np.ones( degree ) , offsets = 1 , shape=(degree+1,degree+1) )
    up_shift = diags( np.ones( degree ) , offsets = -1 , shape=(degree+1,degree+1) )
    D_matrix = diags( np.sqrt( np.arange(0,degree+1)/2.0) , offsets=0, shape=(degree+1,degree+1) )
    return alpha*( down_shift.dot(D_matrix) - D_matrix.dot(up_shift) )

def one_dimensional_mult_op( degree, alpha ):
    #produces the one dimensional multiplication operator in the basis h_n( alpha* x) for n=0,...,degree
    from scipy.sparse import diags
    down_shift = diags( np.ones( degree ) , offsets = 1 , shape=(degree+1,degree+1) )
    up_shift = diags( np.ones( degree ) , offsets = -1 , shape=(degree+1,degree+1) )
    D_matrix = diags( np.sqrt( np.arange(0,degree+1)/2.0) , offsets=0, shape=(degree+1,degree+1) )
    return (down_shift.dot(D_matrix) + D_matrix.dot(up_shift) )/float(alpha)

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    alpha = 1.0
    h_test = hermite_function_series(M=[1,],deg=[20,])
    mu_x = 0.2
    sigma_x = 0.2
    h_test.interpolate( lambda x : np.exp( -(x-mu_x)**2 / (2.0*sigma_x**2) ) )
    x_span = np.linspace(-1,1,100)
    plt.plot( x_span , h_test.evaluate_on_grid([x_span]) ,'b-')
    plt.plot( x_span , np.exp( - (x_span-mu_x)**2 / (2.0*sigma_x**2) ), 'r-')
    plt.grid(True)
    plt.show()
    FP_op_1d = FP_operator(deg=[20,], M=[1,], sigma=[0,], polynomials=[np.array([1,]),])
    dh_test = FP_op_1d.dot( h_test)
    x_span = np.linspace(-1,1,100)
    plt.plot( x_span , dh_test.evaluate_on_grid([x_span]) ,'b-')
    plt.plot( x_span , ( (x_span-mu_x) / (sigma_x**2)) * np.exp( - (x_span-mu_x)**2 / (2.0*sigma_x**2) ), 'r-')
    plt.grid(True)
    plt.show()
