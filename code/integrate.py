import numpy as np

def trap_quad( integrand, bounds, res = None ):
    """ Computes integrals using the n-dimensional trapezoid rule
    args:
        integrand: callable
        bounds: iterable of the form (x_min, x_max, y_min, y_max, ... )
    kwargs:
        res: iterable, of the form (x_res, y_res, ...)
    returns:
        q: float
    Notes:
        The integrand should be of the form f(x,y) for a 2D integrand
        where x and y are each 1D numpy arrays of equal length.
    """

    if res is None:
        res = (len(bounds)/2)*(100,)
    if type(res) == int:
        res = (res,)
    assert( len(res)*2 is len(bounds) )
    def weights_func(r):
        weights_1d = np.ones(r)
        weights_1d[0], weights_1d[-1] = 0.5, 0.5
        return weights_1d
    lower_bnds = [bounds[2*k] for k in range(len(res))]
    upper_bnds = [bounds[2*k+1] for k in range(len(res))]
    grid = np.meshgrid(
            *[np.linspace(lower_bnds[k], upper_bnds[k],res[k])
            for k in range(len(res))])
    dV = reduce( lambda x,y:x*y, map( lambda mn,mx,r: (mx-mn)/float(r-1), lower_bnds, upper_bnds, res ))
    weights = dV*reduce( np.outer, map( weights_func, res) )
    return np.einsum( 'i,i', weights.flatten(), integrand( *grid ).flatten() ) 

if __name__ == '__main__':
    integrand = lambda x: x**2
    res = 50
    bounds = (0,1)
    q = trap_quad( integrand, bounds, res )
    print "Testing 1 dimensional quadrature"
    print "result         = {q}".format(q=q)
    print "correct answer = {a}".format(a=1.0/3.0)
    print "error = {error}".format(error=q-1.0/3.0)

    print "\nTesting 2 dimensional quadrature."
    print "Integrand = ( f(x,y)= exp(-x) )"
    integrand = lambda x,y: np.exp( - x)
    bounds = (-1, 1, -1, 1)
    print "Domain = [{}, {}] x [{}, {}]".format( *bounds)
    res = (100, 100)
    q = trap_quad( integrand, bounds, res )
    a = 2*( np.exp(1) - np.exp(-1) )
    print "result         = {q}".format(q=q)
    print "correct answer = {a}".format(a=a)
    print "error = {error}".format(error=q-a)

    print "\n Testing 3 dimensional quadrature."
    integrand = lambda x,y,z: x+y*z
    bounds = (0,1,0,1,0,1)
    res = (100,100,101)
    q = trap_quad( integrand, bounds, res)
    a = 0.75
    print "result         = {q}".format(q=q)
    print "correct answer = {a}".format(a=a)
    print "error = {error}".format(error=q-a)


