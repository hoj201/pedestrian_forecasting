import numpy as np
from sparse_grid_quad import sparse_grid_quad_2d, sparse_grid_quad_3d, sparse_grid_quad_1d

sigma_x = 0.1
sigma_v = 0.1

class scene():
    """ A class which includes routines for probabilities of a newly observed agent."""
    def __init__( self, curve_ls, V_scale, k_max_theta=6, k_max_vf=10 ):
        """ Initializer

        args:
            curve_ls : list of ndarray, each of shape (2,n) for various n > 0.
            V_scale : tuple of floats, length 2

        kwargs:
            k_max_theta : int
            k_max_vf : int
        """
       
        #Learn the top speed
        def top_speed( curve ):
            fd_width = 4
            n = curve.shape[1]
            u = ( curve[:,fd_width:] - curve[:,:n-fd_width] ) / float(fd_width)
            return ( np.sqrt( u[0]**2 + u[1]**2 ) ).max()

        self.s_max = max( map( top_speed, curve_ls ) )

        #Learn the  agent_classes
        from cluster import get_classes
        alpha_arr, P_of_c, clusters = get_classes( curve_ls, V_scale )
        self.alpha_arr = alpha_arr
        self.P_of_c = P_of_c
        self.V_scale = V_scale 

        #Learn a vector field for each class.  Presumable there are not too many.
        from director_field import trajectories_to_director_field
        X_ls = [] 
        k_max_theta = 4 #max degree of polynomial for angle field
        self.k_max_theta = k_max_theta
        self.num_nl_classes = len(clusters)
        theta_coeffs = np.zeros( (self.num_nl_classes, self.k_max_theta+1, self.k_max_theta+1) )
        for k,cl in enumerate(clusters):
            theta_coeffs[k] = trajectories_to_director_field( cl , V_scale, k_max = self.k_max_theta )
        self.theta_coeffs = theta_coeffs
        self.mu = None
        self.eta = None

    def set_mu(self, mu ):
        self.mu = mu

    def set_eta(self, eta):
        self.eta = eta

    # METHODS
    def director_field(self, k, x, jac=False ):
        """ Returns the value of the vector field for a nonlinear class at a given point.

        args:
            x: ndarray (2,)
            k: int
        
        kwargs:
            jac: If True then returns the Jacobian of the vector-field too.

        returns:
            out1: ndarray (2,)
            out2: ndarray (2,2) if jac==True
        """
        from numpy.polynomial.legendre import legval2d, legder
        theta = legval2d(
                x[0]/self.V_scale[0],
                x[1]/self.V_scale[1],
                self.theta_coeffs[k]
                )
        out1 = np.array( [ np.cos(theta), np.sin(theta) ] )
        if jac:
            dtheta_dx = legval2d(
                        x[0]/self.V_scale[0],
                        x[1]/self.V_scale[1],
                        legder( self.theta_coeffs[k], axis=0)
                        ) / self.V_scale[0]
            dtheta_dy = legval2d(
                        x[0]/self.V_scale[0],
                        x[1]/self.V_scale[1],
                        legder( self.theta_coeffs[k], axis=1)
                        ) / self.V_scale[1]
            out2 = np.zeros( (2,2) )
            out2[0,0] = - out1[1] * dtheta_dx
            out2[0,1] = - out1[1] * dtheta_dy
            out2[1,0] = out1[0] * dtheta_dx
            out2[1,1] = out1[0] * dtheta_dy
            return out1, out2
        return out1

    def director_field_vectorized(self, k, x, jac=False):
        """ Returns the value of the vector field for a nonlinear class at a given point.

        args:
            k (int) : indexes which nonlinear class
            x (numpy.array) :  (2,N)
        
        kwargs:
            jac: If True then returns the Jacobian of the vector-field too.

        returns:
            out1: ndarray (2,N)
            out2: ndarray (2,2,N) if jac==True
        """
        from numpy.polynomial.legendre import legval2d, legder
        theta = legval2d(
                x[0]/self.V_scale[0],
                x[1]/self.V_scale[1],
                self.theta_coeffs[k]
                )
        out1 = np.array( [ np.cos(theta), np.sin(theta) ] )
        if jac:
            dtheta_dx = legval2d(
                        x[0]/self.V_scale[0],
                        x[1]/self.V_scale[1],
                        legder( self.theta_coeffs[k], axis=0)
                        ) / self.V_scale[0]
            dtheta_dy = legval2d(
                        x[0]/self.V_scale[0],
                        x[1]/self.V_scale[1],
                        legder( self.theta_coeffs[k], axis=1)
                        ) / self.V_scale[1]
            N = x.shape[1]
            out2 = np.zeros( (2,2,N) )
            out2[0,0,:] = - out1[1] * dtheta_dx
            out2[0,1,:] = - out1[1] * dtheta_dy
            out2[1,0,:] = out1[0] * dtheta_dx
            out2[1,1,:] = out1[0] * dtheta_dy
            return out1, out2
        return out1

    def P_of_x_given_mu(self, x):
        """ Computes the probability of the true position given just a measurement

        args:
            x : ndarray. Shape=(2,)

        returns:
            out : non negative float
        """
        G = lambda x,s: np.exp( -x**2 / (2*s**2) ) / np.sqrt( 2*np.pi*s**2)
        out = G( x[0]-self.mu[0] , sigma_x )
        out *= G( x[1]-self.mu[1] , sigma_x )
        return out

    def P_of_v_given_eta( self, v ):
        """ Computes the probability density of the true velocity given measurements and a linear-class

        args:
            v : ndarray. Shape(2,)
        
        returns:
            out : positive float.
        """
        G = lambda x,s: np.exp( -x**2 / (2*s**2) ) / np.sqrt( 2*np.pi*s**2)
        out = G( v[0]-self.eta[0] , sigma_v )
        out *= G( v[1]-self.eta[1] , sigma_v )
        return out
        

    def P_of_x_given_nl_class( self, x, k ):
        """ Computes the probability density of the true position, given only the class
        
        args:
            x : ndarray. Shape = (2,)
            k : index of nl class

        returns:
            out : non-negative float
        """
        if not hasattr( self, 'Z_x_given_c'):
            #compute all the partition functions
            Z_ls = []
            for index in range( self.num_nl_classes ):
                def integrand(xy):
                    V = np.polynomial.legendre.legval2d( xy[0]/self.V_scale[0], xy[1]/self.V_scale[1], self.alpha_arr[index] )
                    V -= V.min()
                    return np.exp(-V)
                Z = sparse_grid_quad_2d( integrand, -self.V_scale[0], self.V_scale[0], -self.V_scale[1], self.V_scale[1] )
                Z_ls.append( Z )
            self.Z_x_given_c = tuple( Z_ls )

        V = np.polynomial.legendre.legval2d( x[0] / self.V_scale[0], x[1] / self.V_scale[1], self.alpha_arr[k] )
        V -= V.min()
        return np.exp( -V) / self.Z_x_given_c[k]


    def P_of_linear_given_measurements(self ):
        """ Computes the probability of the linear class given measurements

        args:
        
        returns:
            out : float between 0.0 and 1.0
        """
        #TODO: This over-flows when eta is small
        eta_hat = self.eta / np.sqrt( np.dot(self.eta, self.eta) ) 
        out = 1.0
        gamma_x = 1.0
        gamma_a = 1.0
        for k in range(len(self.P_of_c)-1):
            cos_theta = np.dot( eta_hat, self.director_field(k, self.mu ))
            P_x_given_ck = self.P_of_x_given_nl_class( self.mu, k)
            spatial = np.tanh( gamma_x * P_x_given_ck )
            store = spatial*np.abs( cos_theta)**gamma_a
            out *= 1.0 - store
        return out


    def P_of_nl_class_and_speed_given_measurements( self, k, s):
        """ Computes the probability density of a class and speed given the measurements

        args:
            k (int) : nonlinear class index
            s (float) : speed

        returns:
            out (float): a non-negative float
        """
        if not hasattr( self, 'Z_cs_given_meas' ):
            #COMPUTE THE NORMALIZING CONSTANT
            def integrand( xys, k ):
                s = xys[2]
                xy = xys[0:2]
                v = self.director_field( k, xy )
                v[0] *= s
                v[1] *= s
                out = self.P_of_x_given_mu( xy )
                out *= self.P_of_v_given_eta( v )
                out *= self.P_of_x_given_nl_class( xy , k)
                out *= self.P_of_c[k]
                return out
            x_min = self.mu[0] - 5*sigma_x
            x_max = self.mu[0] + 5*sigma_x
            y_min = self.mu[1] - 5*sigma_x
            y_max = self.mu[1] + 5*sigma_x
            total = 0.
            for k in range( self.num_nl_classes ):
                integrand_k = lambda xys: integrand(xys,k)
                total += sparse_grid_quad_3d( integrand_k, x_min, x_max, y_min, y_max, -self.s_max, self.s_max)
            total /= (1.0 - self.P_of_linear_given_measurements() )
            self.Z_cs_given_meas = total
            
        def integrand( xy ):
            v = self.director_field( k, xy )
            s_filtered = s if np.abs(s) < self.s_max else 0.0
            v[0] *= s_filtered
            v[1] *= s_filtered
            out = self.P_of_x_given_mu( xy )
            out *= self.P_of_v_given_eta( v )
            out *= self.P_of_x_given_nl_class( xy , k)
            out *= self.P_of_c[k]
            return out
        
        x_min = self.mu[0] - 5*sigma_x
        x_max = self.mu[0] + 5*sigma_x
        y_min = self.mu[1] - 5*sigma_x
        y_max = self.mu[1] + 5*sigma_x
        Q = sparse_grid_quad_2d( integrand, x_min, x_max, y_min, y_max )
        return Q / self.Z_cs_given_meas


if __name__ == "__main__":
    print "Testing initializer"
    
    import process_data
    process_data = reload(process_data)
    folder = '../annotations/coupa/video2/'
    fname = folder + 'annotations.txt'
    x_raw,y_raw = process_data.get_trajectories(fname,label="Biker")
    from PIL import Image
    fname = folder + 'reference.jpg'
    im = Image.open(fname)
    width,height = im.size
    print "width = %f, height = %f" % (width,height)
    x_data = map( lambda x: x-width/2 , x_raw )
    y_data = map( lambda x: x-height/2 , y_raw )

    #resize the curves and set V_scale
    x_data = map( lambda x: 2*x/float(width) , x_data )
    y_data = map( lambda x: 2*x/float(width) , y_data ) #Note a typo.  We resize by the same factor along both directions
    V_scale = (1.0, height / float(width) )
    curve_ls = [ np.vstack([x,y]) for (x,y) in zip( x_data, y_data ) ]
    from sklearn.cross_validation import train_test_split
    train_set, test_set = train_test_split( curve_ls, random_state = 0 )

    coupa_scene = scene( train_set, V_scale )
    coupa_scene.set_mu( np.zeros(2) )
    coupa_scene.set_eta( np.ones(2) )
    print "mu = "
    print coupa_scene.mu
    print "eta = "
    print coupa_scene.eta

    x = np.zeros(2)

    from time import time


    print "Testing P( linear | mu,eta) runs"
    coupa_scene.set_mu( [0.8, 0.4])
    coupa_scene.set_eta( np.array( [1.0,-1.0] ) )
    t0 = time()
    res = coupa_scene.P_of_linear_given_measurements()
    print "result = %f" % res
    print "Should be nearly 1"
    print "CPU time = %f \n" % (time()-t0)


    coupa_scene.set_mu( np.array([0.0, 0.4]) )
    coupa_scene.set_eta( np.array( [1.0,0.0] ) )
    t0 = time()
    res = coupa_scene.P_of_linear_given_measurements()
    print "result = %f" % res
    print "Should be positive but closer to 0 than 1"
    print "CPU time = %f \n" % (time()-t0)


    print "Testing if P(c,s|mu,eta) runs"
    coupa_scene.mu = np.array( [0.0, 0.6] )
    coupa_scene.eta = coupa_scene.director_field(0, coupa_scene.mu )
    t0 = time()
    res = coupa_scene.P_of_nl_class_and_speed_given_measurements( 0, 1.0 )
    print "result = "
    print res
    print "CPU time = %f \n" % (time()-t0)


    print "Testing if P(linear | mu,eta) + \sum_k \int P(ck,s|mu,eta)ds = 1"
    I = 0.
    s_min = -coupa_scene.s_max
    s_max = coupa_scene.s_max
    res = 100.0
    ds = (s_max - s_min ) / (res-1)
    for k in range( coupa_scene.num_nl_classes ):
        integrand = lambda s: coupa_scene.P_of_nl_class_and_speed_given_measurements( k, s)
        I += sum( [ integrand(s)*ds for s in np.linspace(s_min, s_max, res ) ] ) 
    I += coupa_scene.P_of_linear_given_measurements()
    print "Sum = %f\n" % I


    from matplotlib import pyplot as plt
    alpha = coupa_scene.alpha_arr
    V_scale = coupa_scene.V_scale
    X_grid, Y_grid = np.meshgrid( np.linspace(-V_scale[0], V_scale[0], 50),
            np.linspace(-V_scale[1], V_scale[1], 50) )
    from numpy.polynomial.legendre import legval2d
    fig, ax_arr = plt.subplots( coupa_scene.num_nl_classes)
    for k in range( coupa_scene.num_nl_classes ):
        Z_grid = legval2d( X_grid/V_scale[0], Y_grid/V_scale[1], alpha[k])
        Z_grid -= Z_grid.min()
        cs = ax_arr[k].contourf( X_grid, Y_grid, Z_grid , 50 )
    plt.show()


