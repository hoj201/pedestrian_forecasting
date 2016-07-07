import numpy as np

def hermite_function(x, deg=20):
    # computes h_n(x) for n = 0,...,deg
    out = np.zeros( (deg + 2 , x.size) )
    out[0,:] = np.pi**-0.25 * np.exp( - x**2 / 2.0)
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
    def __init__(self, coeffs = None , dim=None , M=1, deg=20):
        self.M = M
        self.deg = deg
        if(coeffs is None):
            assert( dim > 0 )
            self.dim = dim
            self.coeffs = np.array( [deg+2]*dim )
        else:
            if( dim is None):
                self.dim = len( coeffs.shape )
            else:
                self.dim = dim
            assert( (deg+2)**self.dim == coeffs.size )
            self.coeffs = coeffs.reshape( [deg+2]*dim )

    def set_coeffs(self , x ):
        if( len(x.shape)==1):
            #x is a flat array.  Reshape it appropriately
            assert( x.size == self.coeffs.size )
            self.coeffs = x.reshape( [self.deg+2]*self.dim )
            return 0
        assert( x.shape == self.coeffs.shape )
        self.coeffs = x
        return 0


    def interpolate_function(self, f):
        #computes the hermite series which interpolates f on [-M,M]^d.
        #Output: coefficient array a[n] where f(x) = a[n] h_n(x*alpha)
        #where alpha = sqrt(2*deg)/M
        from numpy import meshgrid,sqrt
        from numpy.polynomial.hermite import hermgauss
        gamma,w = hermgauss( self.deg )
        h = hermite_function( gamma , self.deg )
        C = ( h**2).sum(axis=0)
        alpha = sqrt( 2*self.deg) / self.M
        f_at_gamma = f( *meshgrid( *[gamma/alpha ]*self.dim ) )
        h_list = []
        C_inverse_list = []
        f_index = []
        output_index = []
        n=0
        i=self.dim
        for _ in range(self.dim):
            h_list.append( h )
            h_list.append( [n,i] )
            C_inverse_list.append( C**-1 )
            C_inverse_list.append( [i] )
            f_index.append( i )
            output_index.append(n)
            i += 1
            n += 1
        arg = h_list + C_inverse_list + [f_at_gamma, f_index, output_index]
        self.coeffs = np.einsum( *arg )
        return 0

    def evaluate_on_uniform_grid( self , res=20 ):
        #given a hermite series a[m,n] we evaluate it on a grid
        #Output: array f[i] = \sum_m a[m] h_m( x_i * alpha )
        #where alpha = sqrt(2*res)/M
        M=self.M
        alpha = np.sqrt( 2*self.deg )/M
        from numpy import linspace
        h = hermite_function( linspace(-M,M,res)*alpha , deg=self.deg )
        #In 2D this call looks like
        #np.einsum(a,[m1,m2],h,[m1,i1], h, [m2,i2] , [i1,i2] )
        m_list = []
        i_list = []
        h_list = []
        m = 0
        i = self.dim
        for _ in range(self.dim):
            m_list.append(m)
            i_list.append(i)
            h_list += [ h , [m,i] ]
            i += 1
            m += 1
        arg = [ self.coeffs , m_list ] + h_list + [i_list]
        from numpy import einsum
        return einsum( *arg )

    def get_uniform_grid( self, res=20):
        from numpy import meshgrid,linspace
        return meshgrid( *[linspace( -self.M , self.M , res )]*self.dim )
    
    def __add__( self , other):
        assert( type(other) == type(self) )
        assert( self.M == other.M )
        assert( self.deg == other.deg )
        assert( self.dim == other.dim )
        return hermite_function_series( coeffs = self.coeffs + other.coeffs , M = self.M , deg = self.deg, dim =self.dim )

    def __mul__( self , x ):
        #scalar multiplication
        return hermite_function_series( coeffs = x*self.coeffs , M = self.M , deg = self.deg , dim = self.dim )

    def __lmul__(self , x ):
        return hermite_function_series( coeffs = x*self.coeffs , M = self.M , deg = self.deg , dim = self.dim )

    def __rmul__(self , x ):
        return hermite_function_series( coeffs = x*self.coeffs , M = self.M , deg = self.deg , dim = self.dim )

class Lie_derivative:
    #Produces a Lie derivative operator for a polynomial vector field
    def __init__(self, polynomials=None , M=1, dim=1, deg=20):
        self.dim = dim
        self.deg = deg
        self.M = M
        from scipy import sparse
        if( polynomials is None ):
            self.op = sparse.dia_matrix( ( (deg+2)**dim , (deg+2)**dim ) )
        else:
            dim = len(polynomials)
            self.dim = dim
            d = np.sqrt( np.arange(1,deg+2) )
            alpha = np.sqrt(2*deg)/M
            diff_by_x = alpha*sparse.diags( [-d,d] , offsets=[1,-1] , shape=(deg+2,deg+2) )
            mult_by_x = (alpha**-1)*sparse.diags( [d,d] , offsets=[1,-1] , shape=(deg+2,deg+2) )
            multiplication_operators = []
            derivative_op = []
            for k in range(dim):
                for i in range(dim):
                    if(i==k):
                        store_m = mult_by_x
                        store_d = diff_by_x
                    else:
                        store_m = sparse.eye( deg + 2 )
                        store_d = sparse.eye( deg + 2 )
                    if( i == 0):
                        mult_by_kth = store_m.copy()
                        diff_by_kth = store_d.copy()
                    else:
                        mult_by_kth = sparse.kron( store_m, mult_by_kth )
                        diff_by_kth = sparse.kron( store_d, diff_by_kth )
                multiplication_operators.append( mult_by_kth )
                derivative_op.append( diff_by_kth )
            self.op = sparse.dia_matrix( ( (deg+2)**dim , (deg+2)**dim ) )
            for k in range(dim):
                mult_by_poly = eval_nd_poly( polynomials[k] , multiplication_operators )
                self.op += 0.5*derivative_op[k].dot( mult_by_poly ) + 0.5*mult_by_poly.dot( derivative_op[k] )
    def __add__(self, other ):
        assert( type(self) == type(other) )
        assert( self.M == other.M and self.dim == other.dim and self.deg==other.deg)
        out = Lie_derivative( M=self.M, dim=self.dim, deg=self.deg )
        out.op = self.op + other.op
        return out

    def dot( self, h_series ):
        #applies the Lie derivative Operatot to a hermite series
        assert( self.M == h_series.M )
        assert( self.deg == h_series.deg )
        assert( self.dim == h_series.dim )
        coeffs = self.op.dot( h_series.coeffs.flatten() ).reshape( [self.deg+2]*self.dim )
        return hermite_function_series( coeffs = coeffs, M = self.M , dim=self.dim, deg=self.deg )

    def advect( self, h_series, t , rtol = None ):
        #evolves h_series by time t according to the Schrodinger equation
        assert( self.M == h_series.M )
        assert( self.deg == h_series.deg )
        assert( self.dim == h_series.dim )
        x0 = h_series.coeffs.flatten()
        from scipy.integrate import odeint
        x_arr = odeint( lambda x,t: self.op.dot(x) , x0 , np.array([0,t]) )
        return hermite_function_series( coeffs=x_arr[1].reshape( [self.deg+2]*self.dim ) , dim=self.dim  ,M=self.M,deg=self.deg)

    def cayley_step(self, h_series , dt ):
        #evolves h_series by cayley(A dt) = ( I-dt*A)^{-1} (I+A*dt).
        #where A = self.op
        assert( [self.dim,self.M,self.deg] == [h_series.dim,h_series.M,h_series.deg] )
        from scipy import sparse
        I = sparse.eye( (self.deg+2)**self.dim )
        x = h_series.coeffs.flatten()
        x += 0.5*dt*self.op.dot( x )
        from scipy.sparse.linalg import spsolve
        y = spsolve( I - 0.5*dt*self.op , x )
        return hermite_function_series( coeffs = y , M=self.M,deg=self.deg,dim=self.dim)




def horners( a , x ):
    n = len(a)
    from scipy import sparse
    Id = sparse.eye( x.shape[0] )
    b = a[n-1]*Id
    tol = 0.01
    for k in range( n-2,-1,-1):
        if( np.abs(a[k]) < tol ):
            b = b.dot(x)
        else:
            b = a[k]*Id + b.dot( x )
    return b

def eval_nd_poly( a , x ):
    if( len(a.shape) > 1 ):
        n = a.shape[0]
        b = eval_nd_poly( a[n-1,:] , x[1:len(x)] )
        for k in range( n-2, -1, -1 ):
            b = eval_nd_poly( a[k,:] , x[1:len(x)] ) + x[0].dot(b)
        return b
    return horners( a , x[0] )

