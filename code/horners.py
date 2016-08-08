import numpy as np

def horners( a , x, mul = lambda a,b:a*b ):
    n = len(a)
    from numbers import Number
    from scipy.sparse import spmatrix
    if isinstance(x,Number):
        Id = 1.0
    elif isinstance(x,spmatrix):
        from scipy.sparse import eye as speye
        Id = speye( x.shape[0] )
    else:
        Id = np.eye( x.shape[0] )
    b = a[n-1]*Id
    tol = 1e-8
    for k in range( n-2,-1,-1):
        if( np.abs(a[k]) < tol ):
            b = mul(b,x)
        else:
            b = a[k]*Id + mul(b, x )
    return b

def eval_nd_poly( a , x , mul = lambda a,b: a*b ):
    if( len(a.shape) > 1 ):
        n = a.shape[0]
        b = eval_nd_poly( a[n-1,:] , x[1:len(x)] )
        for k in range( n-2, -1, -1 ):
            b = eval_nd_poly( a[k,:] , x[1:len(x)] ) + mul( x[0], b )
        return b
    return horners( a , x[0] , mul=mul )

if __name__ == '__main__':
    print "Test: x=2,y=13, so x^2 + 2*x*y = 56"
    coeffs = np.zeros( (3,2) , dtype=int )
    coeffs[2,0] = 1
    coeffs[1,1] = 2
    print "output = %d " % eval_nd_poly( coeffs , [2,13] )
    
    print "\nTest: x = [[0,1],[-1,0]], y=[ [2,0],[0,2] ]"
    x = np.array( [[0,1],[-1,0]], dtype=int)
    y = np.array( [[2,0],[0,2]], dtype=int)
    print "x^2 + 2*x y = "
    print x.dot(x) + 2*x.dot(y)
    output = eval_nd_poly( coeffs, [x,y] , mul=lambda a,b: a.dot(b) )
    print "output = \n" + str(output)

    x = np.random.randn()
    y = np.random.randn()
    z = np.random.randn()
    print "\nTest: \n x=%f \n y=%f \n z=%f" % (x,y,z)
    correct = x**2 + np.pi*x*y*z + z**2 + x**2 * z
    print "x^2 + pi*xyz + z^2 + x^2 z = %f" % correct
    coeffs = np.zeros( (3,2,3) )
    coeffs[2,0,0] = 1.0
    coeffs[1,1,1] = np.pi
    coeffs[0,0,2] = 1.0
    coeffs[2,0,1] = 1.0
    print "output = %f" % eval_nd_poly( coeffs, [x,y,z] )
