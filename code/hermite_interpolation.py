import numpy as np
def hermite_function(x,max_degree=20):
    # computes h_n(x) for n = 0,...,max_degree
    out = np.zeros( (max_degree + 2 , x.size) )
    out[0,:] = np.pi**-0.25 * np.exp( - x**2 / 2.0)
    out[1,:] = np.sqrt(2)*x*out[0,:]
    for n in range(2,max_degree+1):
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

def interpolation_2d(f , M=1, deg=20 ):
    #computes the hermite-series which interpolates the function f supported on [-M,M]^2.
    #Output: coefficient array a[n] where f(x,y) = a[m,n] h_m( x * alpha ) h_n( y*alpha)
    #where alpha = sqrt(2*deg)/M
    gamma,w = np.polynomial.hermite.hermgauss( deg )
    h = hermite_function( gamma , deg )
    C = ( h**2).sum(axis=0)
    x_gamma,y_gamma = np.meshgrid( gamma , gamma )
    alpha = np.sqrt( 2*deg) / M
    return np.einsum('mi,nj,i,j,ij->mn',h , h , C**-1, C**-1, f( x_gamma/alpha,y_gamma/alpha ))
   
def interpolation_nd(f , M=1, deg=20):
    #computes the hermite series which interpolates f on [-M,M]^d.
    #Output: coefficient array a[n] where f(x) = a[n] h_n(x*alpha)
    #where alpha = sqrt(2*deg)/M
    gamma,w = np.polynomial.hermite.hermgauss( deg )
    h = hermite_function( gamma , deg )
    C = ( h**2).sum(axis=0)
    dim = f.func_code.co_argcount #Dimension of system
    alpha = np.sqrt( 2*deg) / M
    f_at_gamma = f( *np.meshgrid( *[gamma/alpha ]*dim ) )
    h_list = []
    C_inverse_list = []
    f_index = []
    output_index = []
    n=0
    i=dim
    for _ in range(dim):
        h_list.append( h )
        h_list.append( [n,i] )
        C_inverse_list.append( C**-1 )
        C_inverse_list.append( [i] )
        f_index.append( i )
        output_index.append(n)
        i += 1
        n += 1
    arg = h_list + C_inverse_list + [f_at_gamma, f_index, output_index]
    return np.einsum( *arg )
