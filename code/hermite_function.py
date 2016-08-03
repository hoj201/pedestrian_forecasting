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
    def __init__(self, coeffs = None , M=(1,), deg=(20,)):
        self.M = M
        self.deg = deg
        self.dim = len(M)
        shape_tuple = map( lambda x: x+2 , deg )
        if(coeffs is None):
            self.coeffs = np.zeros( shape_tuple )
        else:
            self.coeffs = coeffs.reshape( shape_tuple )

    def set_coeffs(self , x ):
        self.coeffs = x.reshape( self.coeffs.shape )
        return 0

    def interpolate_function(self, f):
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

    def get_uniform_grid( self, res=20):
        from numpy import meshgrid,linspace
        return meshgrid( * map( lambda m: linspace(-m,m,res) , self.M ) )

    def __add__( self , other):
        assert( type(other) == type(self) )
        assert( self.M == other.M )
        assert( self.deg == other.deg )
        return hermite_function_series( coeffs = self.coeffs + other.coeffs, M = self.M, deg = self.deg)

    def __mul__( self , x ):
        #scalar multiplication
        return hermite_function_series( coeffs = x*self.coeffs, M = self.M, deg = self.deg )

    def __lmul__(self , x ):
        return hermite_function_series( coeffs = x*self.coeffs, M = self.M, deg = self.deg )

    def __rmul__(self , x ):
        return hermite_function_series( coeffs = x*self.coeffs, M = self.M, deg = self.deg )

class FP_operator:
    #Produces a Fokker-Planck operator for densities with respect to polynomial vector fields and Gaussian noise
    def __init__(self, polynomials=None , M=(1,), deg=(20,) , sigma = None ):
        assert( len(deg) == len(M) )
        self.deg = deg
        self.M = M
        from scipy import sparse
        n_modes = reduce( lambda x,y: x*(y+2) , deg , 1 )
        print n_modes
        self.op = sparse.dia_matrix( ( n_modes , n_modes ) )
        if( polynomials != None ):
            assert( len( polynomials) == len(self.deg) )
            alpha = np.sqrt(2*np.array(deg))/np.array(M)
            multiplication_operators = []
            derivative_op = []
            for k in range(len(self.deg) ):
                d = np.sqrt( np.arange(1,deg[k]+2) )
                diff_1d = alpha[k]*sparse.diags( [-d,d] , offsets=[1,-1] , shape=(deg[k]+2,deg[k]+2) )
                mult_1d = (alpha[k]**-1)*sparse.diags( [d,d] , offsets=[1,-1] , shape=(deg[k]+2,deg[k]+2) )
                for i in range(len(self.deg)):
                    if(i==k):
                        store_m = diff_1d
                        store_d = mult_1d
                    else:
                        store_m = sparse.eye( deg[i] + 2 )
                        store_d = sparse.eye( deg[i] + 2 )
                    if( i == 0):
                        mult_by_kth = store_m.copy()
                        diff_by_kth = store_d.copy()
                    else:
                        mult_by_kth = sparse.kron( store_m, mult_by_kth )
                        diff_by_kth = sparse.kron( store_d, diff_by_kth )
                multiplication_operators.append( mult_by_kth )
                derivative_op.append( diff_by_kth )
            for k in range( len(self.deg)):
                mult_by_poly = eval_nd_poly( polynomials[k] , multiplication_operators )
                self.op -= derivative_op[k].dot( mult_by_poly )
            if sigma != None:
                assert( len(sigma) == len(M) )
                for k in range( len(self.deg) ):
                    self.op += sigma[k] * derivative_op[k].dot( derivative_op[k] )
 
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

    def advect( self, h_series, t , rtol = None ):
        #evolves h_series by time t according to the Schrodinger equation
        assert( self.M == h_series.M )
        assert( self.deg == h_series.deg )
        x0 = h_series.coeffs.flatten()
        from scipy.integrate import odeint
        x_arr = odeint( lambda x,t: self.op.dot(x) , x0 , np.array([0,t]) )
        return hermite_function_series( coeffs=x_arr[1].reshape( h_series.coeffs.shape ) , dim=self.dim  ,M=self.M,deg=self.deg)

    def cayley_step(self, h_series , dt ):
        #evolves h_series by cayley(A dt) = ( I-dt*A)^{-1} (I+A*dt).
        #where A = self.op
        assert( self.M == h_series.M )
        assert( self.deg == h_series.deg )
        from scipy import sparse
        I = sparse.eye( 1) #TODO: Fix this if your advection routine failes
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

