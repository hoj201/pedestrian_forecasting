import numpy as np

sigma_x = 0.1
sigma_v = 0.1

from pandas import read_csv
sparse_grid_nodes_3d = read_csv('./sparse_grid_data/points_3d.dat', sep=',', header=None).values
sparse_grid_weights_3d = read_csv('./sparse_grid_data/weights_3d.dat', sep=',', header=None).values.flatten()
sparse_grid_nodes_2d = read_csv('./sparse_grid_data/points_2d.dat', sep=',', header=None).values
sparse_grid_weights_2d = read_csv('./sparse_grid_data/weights_2d.dat', sep=',', header=None).values.flatten()


def sparse_grid_quad_2d( integrand, x_min, x_max, y_min, y_max ):
    #Computes the sparse grid quadrature of the integrand over the box [x_min,x_max] x [y_min, y_max]
    transformed_nodes = np.zeros_like( sparse_grid_nodes_2d )
    transformed_nodes[:,0] = sparse_grid_nodes_2d[:,0]*(x_max-x_min) + x_min
    transformed_nodes[:,1] = sparse_grid_nodes_2d[:,1]*(y_max-y_min) + y_min
    vals_on_nodes = integrand( transformed_nodes.transpose() )
    Vol = (x_max - x_min ) * (y_max - y_min )
    return np.dot( vals_on_nodes , sparse_grid_weights_2d ) * Vol

def sparse_grid_quad_3d( integrand, x_min, x_max, y_min, y_max, z_min, z_max ):
    #Computes the sparse grid quadrature of the integrand over the box [x_min,x_max] x [y_min, y_max]
    transformed_nodes = np.zeros_like( sparse_grid_nodes_3d )
    transformed_nodes[:,0] = sparse_grid_nodes_3d[:,0]*(x_max-x_min) + x_min
    transformed_nodes[:,1] = sparse_grid_nodes_3d[:,1]*(y_max-y_min) + y_min
    transformed_nodes[:,2] = sparse_grid_nodes_3d[:,2]*(z_max-z_min) + z_min
    vals_on_nodes = integrand( transformed_nodes.transpose() )
    Vol = ( x_max-x_min ) * ( y_max - y_min) * (z_max - z_min )
    return np.dot( vals_on_nodes , sparse_grid_weights_3d ) * Vol


def memoize(f):
    memo = {}
    def helper(x):
        if x.tostring() not in memo:
            memo[x.tostring()] = f(x)
        return memo[x.tostring()]
    return helper


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
        
        #First learn the posible agent_classes
        from cluster import get_classes
        alpha_arr, P_of_c, clusters = get_classes( curve_ls, V_scale )
        self.alpha_arr = alpha_arr
        self.P_of_c = P_of_c
        self.V_scale = V_scale 
        from matplotlib import pyplot as plt
        fig, ax_arr = plt.subplots( len(clusters) , figsize = (5,8) )
        X_grid, Y_grid = np.meshgrid( np.linspace( -V_scale[0], V_scale[0], 50),
                np.linspace( -V_scale[1], V_scale[1], 50))
        for k,cl in enumerate(clusters):
            Z_grid = np.polynomial.legendre.legval2d( X_grid / V_scale[0], Y_grid / V_scale[1], alpha_arr[k] )
            ax_arr[k].contourf( X_grid, Y_grid, Z_grid, cmap='viridis' )
            for xy in cl:
                ax_arr[k].plot( xy[0], xy[1], 'w-' )
        plt.show()

        #Learn a vector field for each class.  Presumable there are not too many.
        from director_field import trajectories_to_director_field
        X_ls = [] 
        k_max_theta = 4 #max degree of polynomial for angle field
        num_nonlinear_classes = len(clusters)
        theta_coeffs = np.zeros( (num_nonlinear_classes, k_max_theta+1, k_max_theta+1) )
        for k,cl in enumerate(clusters):
            theta_coeffs[k] = trajectories_to_director_field( cl , V_scale, k_max = k_max_theta )
        self.theta_coeffs = theta_coeffs
        self.mu = None
        self.eta = None

    def set_mu(self, mu ):
        self.mu = mu

    def set_eta(self, eta):
        self.eta = eta

    # METHODS
    def director_field(self, k, x, y):
        theta = np.polynomial.legendre.legval2d(
                x/self.V_scale[0],
                y/self.V_scale[1],
                self.theta_coeffs[k]
                )
        return np.array( [ np.cos(theta), np.sin(theta) ] )


    def director_field_vectorized( self, k, xy ):
        """ Returns the director field at a variety of points

        args:
            k (int):
            xy (ndarray): shape = (2,n)

        returns:
            out (ndarray): shape = (2,n)
        """
        x = xy[0]
        y = xy[1]
        theta = np.polynomial.legendre.legval2d(
                x/self.V_scale[0],
                y/self.V_scale[1],
                self.theta_coeffs[k]
                )
        return np.vstack( [ np.cos(theta), np.sin(theta) ] )
        

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
            for index in range( len( self.P_of_c )-1 ):
                def integrand(xy):
                    V = np.polynomial.legendre.legval2d( xy[0]/self.V_scale[0], xy[1]/self.V_scale[1], self.alpha_arr[index] )
                    return np.exp(-V)
                Z = sparse_grid_quad_2d( integrand, -self.V_scale[0], self.V_scale[0], -self.V_scale[1], self.V_scale[1] )
                Z_ls.append( Z )
            self.Z_x_given_c = tuple( Z_ls )

        V = np.polynomial.legendre.legval2d( x[0] / self.V_scale[0], x[1] / self.V_scale[1], self.alpha_arr[k] )
        return np.exp( -V) / self.Z_x_given_c[k]


    def P_of_x_given_measurements_nonlinear_class_speed( self, x, k , speed ):
        """ Computes the probability density of true position for a given class speed and measurements

        args:
            x : ndarray.  Shape = (2,N_points)
            k : class index
            speed : float

        returns:
            out : ndarray (N_points,).  dtype = float, positive
        """
        global sigma_x, sigma_v
        from scipy.integrate import dblquad
        ss = lambda x: np.dot(x,x)
        #If measured speed is nearly zero we assume no motion
        if np.sqrt(ss(self.eta)) <= 1e-5:
            return P_of_x_given_mu( x )

        eta_hat = self.eta / np.sqrt( ss(self.eta))
        def integrand_sparse(xy): #This swapping of variables is deliberate
            v = self.director_field(k, xy[0], xy[1] ) 
            out = np.exp( - (v[0]-eta_hat[0])**2 / (2*sigma_v**2) ) / np.sqrt( 2*np.pi*sigma_v**2 )
            out *= np.exp( - (v[1]-eta_hat[1])**2 / (2*sigma_v**2) ) / np.sqrt( 2*np.pi*sigma_v**2 )
            out *= np.exp( - ( xy[0] - self.mu[0] )**2 / (2*sigma_x**2) )  / np.sqrt(2*np.pi*sigma_x**2 ) 
            out *= np.exp( - ( xy[1] - self.mu[1] )**2 / (2*sigma_x**2) )  / np.sqrt(2*np.pi*sigma_x**2 ) 
            return out

        x_min = self.mu[0] - 7*sigma_x
        x_max = self.mu[0] + 7*sigma_x
        y_min = self.mu[1] - 7*sigma_x
        y_max = self.mu[1] + 7*sigma_x
        Z_sg = sparse_grid_quad_2d( integrand_sparse, x_min, x_max, y_min, y_max )
        return integrand_sparse(x)/ Z_sg

    #TODO: This runs and appears to behave right.  Has not been checked for accu.
    def P_of_nonlinear_class_and_speed_given_measurements( self, class_index, speed ):
        """ Computes the probabilities of class and speed s given measurements

        args:
            class_index: int
            speed : float

        returns:
            out : float between 0 and 1
        """
        def integrand_sparse(xy):
            v = speed * self.director_field(class_index, xy[0], xy[1])
            out = np.exp( -(xy[0]-self.mu[0] )**2 / (2*sigma_x**2) )
            out *= np.exp( -(xy[1]-self.mu[1] )**2 / (2*sigma_x**2) )
            out *= np.exp( -(v[0]-self.eta[0] )**2 / (2*sigma_v**2) )
            out *= np.exp( -(v[1]-self.eta[1] )**2 / (2*sigma_v**2) )
            return out

        x_min = self.mu[0] - 7*sigma_x
        x_max = self.mu[0] + 7*sigma_x
        y_min = self.mu[1] - 7*sigma_x
        y_max = self.mu[1] + 7*sigma_x
        numerator = sparse_grid_quad_2d( integrand_sparse, x_min, x_max, y_min, y_max )

        if not hasattr( self, 'Z_nl_c_given_measurements' ):
            Z = 0.0
            x_min = self.mu[0] - 7*sigma_x
            x_max = self.mu[0] + 7*sigma_x
            y_min = self.mu[1] - 7*sigma_x
            y_max = self.mu[1] + 7*sigma_x
            s_max = np.sqrt( np.dot(self.eta, self.eta) ) + np.sqrt( np.dot(self.eta,self.eta) + (7*sigma_x)**2 )
            s_min = -s_max
            for k in range( len( self.P_of_c) -1):
                def Z_integrand( xys ):
                    #make this evaluate on a Nx3 array
                    x = xys[0]
                    y = xys[1]
                    s = xys[2]
                    v = np.einsum('j,ij->ij', s, self.director_field( k, x, y) )
                    out = np.exp( -(x - self.mu[0])**2 / (2*sigma_x**2) )
                    out = np.exp( -(y - self.mu[1])**2 / (2*sigma_x**2) )
                    out = np.exp( -(v[0] - self.eta[0])**2 / (2*sigma_v**2) )
                    out = np.exp( -(v[1] - self.eta[1])**2 / (2*sigma_v**2) )
                    return out
                #Using Clenshaw-Curtis quadrature rule
                #evaluate on nodes
                node_vals = Z_integrand( sparse_grid_nodes_3d.transpose() ).flatten()
                Z += sparse_grid_quad_3d( Z_integrand, x_min, x_max, y_min, y_max, s_min, s_max )
            #Now we divide by (1-P( Linear | measurements) ) #TODO:  Overflow error when Linear predictor is likely
            Z /= 1.0 - self.P_of_linear_given_measurements()
            self.Z_nl_c_given_measurements = Z
        return numerator / self.Z_nl_c_given_measurements

    def P_of_future_position_given_linear_class_and_measurements( self, xT, T ):
        """ Computes the probability density at xT at time T under linear motion

        args:
            xT : ndarray. Shape (2,)
            T  : float. 

        returns:
            out : positive float.
        """
        global sigma_x, sigma_v
        #1/(2pi sig_x sig_v) \int G_{sig_x}( xT - Tv - mu ) + G_{sig_v}( v - eta) dv
        from scipy.integrate import dblquad
        G = lambda x,s: np.exp( -x**2 / (2*s**2) ) / np.sqrt(2*np.pi*s**2)
        def integrand_sparse(uv):
            out = G( xT[0]-T*uv[0] - self.mu[0] , sigma_x)
            out *= G( xT[1]-T*uv[1] - self.mu[1] , sigma_x)
            out *= G( self.eta[0] - uv[0] , sigma_v)
            out *= G( self.eta[1] - uv[1] , sigma_v)
            return out
        u_min = self.eta[0]+7*sigma_v
        u_max = self.eta[0]-7*sigma_v
        v_min = self.eta[1]+7*sigma_v
        v_max = self.eta[1]-7*sigma_v
        out = sparse_grid_quad_2d( integrand_sparse, u_min, u_max, v_min, v_max )
        return out

    def P_of_linear_given_measurements(self ):
        """ Computes the probability of the linear class given measurements

        args:
        
        returns:
            out : float between 0.0 and 1.0
        """
        eta_hat = self.eta / np.sqrt( np.dot(self.eta, self.eta) ) #TODO: This over-flows when eta is small
        x,y = list( self.mu )
        out = 1.0
        gamma_x = 1.0
        gamma_a = 1.0
        for k in range(len(self.P_of_c)-1):
            store = np.abs(np.cos( np.dot( eta_hat, self.director_field(k,x,y) ) ))**gamma_a
            store *= np.tanh( gamma_x * self.P_of_x_given_nl_class( self.mu, k) )
            out *= 1.0 - store
        return out


    #TODO: FINISH CODING THIS
    def P_of_c_and_s_given_measurements( self, k, s):
        """ Computes the probability density of a class and speed given the measurements

        args:
            k (int) : nonlinear class index
            s (float) : speed

        returns:
            out (float): a non-negative float
        """
        if not hasattr( self, 'Z_cs_given_meas' ):
            #COMPUTE THE NORMALIZING CONSTANT
            print "Still need to code Z_cs_given_meas"


        G = lambda x,sigma: np.exp( -np.dot(x,x)) / (2*sigma**2)
        def integrand(y,x): #Swap is deliberate.  See scipy.integrate.dblquad docs
            global sigma_x
            xy = np.array([x,y])
            out = G( xy-self.mu, sigma_x )
            out *= G( s*self.director_field(k,x,y) - self.eta )
            return out
        
        x_lower = self.mu[0] - 7*sigma_x
        x_upper = self.mu[0] + 7*sigma_x
        y_lower = lambda x: mu[1] - np.sqrt( (7*sigma_x)**2 - (self.mu[0]-x)**2 )
        y_upper = lambda x: mu[1] + np.sqrt( (7*sigma_x)**2 - (self.mu[0]-x)**2 )
        
        from scipy.integrate import dblquad
        integral, abs_err = dblquad( integrand, x_lower, x_upper, y_lower, y_upper , epsabs = 1e-5, epsrel=1e-5)
        if abs_err > 1e-5:
            print "Warning in P_of_c_and_s_given_measurements"
        return integral / Z #MAKE Z A STATIC VARIABLE


    #TODO: FINISH CODING THIS
    def P_of_x_given_measurements_and_linear_class( self, x ):
        """ Computes the probability density of the true location given measurements and a linear-class

        args:
            x : ndarray. Shape(2,)
        
        returns:
            out : positive float.
        """
        out = self.P_of_x_given_mu( x) / P_of_c[-1]
        #Now use the rule of total probability conditioning on class and speed
        for k in range( len(P_of_c)-1 ):
            Px_given_cs = self.P_of_x_given_measurement_nonlinear_class_speed( x, k, 1.0 ) #Note: Px_given_class_speed does not depend on speed in this implementation 
            integrand = lambda s: P_of_c_and_s_given_measurements(k,s)
            a = mag_eta - 7*sigma_v
            b = mag_eta + 7*sigma_v
            from scipy.integrate import quad
            integral, abs_err = quad( integrand, a, b )
            if abs_err > 1e-5:
                print "Warning in P_of_x_given_measurements_and_linear_class"
            out -= integral / P_of_c[-1]
        return out

    def P_of_x_given_mu( x):
        """ Computes the probability of the true position given just a measurement

        args:
            x : ndarray. Shape=(2,)

        returns:
            out : non negative float
        """
        return np.exp( -np.dot(x-self.mu, x-self.mu)) / (2*sigma_x**2)

    #TODO:  Check if you actually need this.  I think you do not.
    def P_of_v_given_measurements_and_linear_class( self, v ):
        """ Computes the probability density of the true velocity given measurements and a linear-class

        args:
            v : ndarray. Shape(2,)
        
        returns:
            out : positive float.
        """
        #TODO: Code it
        return -1

@memoize
def partition_function( alpha, V_scale ):
    """
    Computes the partition function for each potential

    args:
        alpha (numpy.ndarray): coefficients for legval2d for each class, shape=(n,k_max+1,k_max+1)

    returns:
        Z (numpy.ndarray): partition function for each class, shape=(n,)
    """
    res = 50
    x_grid, y_grid = np.meshgrid( np.linspace( -V_scale[0], V_scale[0],res),
            np.linspace(-V_scale[1], V_scale[1], res ) )
    n_classes = alpha.shape[0]
    Area = 4*V_scale[0]*V_scale[1]
    V = np.polynomial.legendre.legval2d( x_grid / V_scale[0], y_grid / V_scale[1], np.transpose( alpha, (1,2,0)  ) )
    return Area*np.exp(-V).mean(axis=(1,2))


if __name__ == "__main__":
    """
    print "Testing partition function"
    k_max = 6
    alpha = np.zeros( (2, k_max+1, k_max+1 ) )
    alpha[0,1,0] = 1.0
    V_scale = (1.0, 1.0 )
    Z = partition_function( alpha, V_scale )
    print "Z = %f" % Z[0]
    print "Correct answer is Z=4"
    """
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

    from matplotlib import pyplot as plt
    for k in range(len(x_data) ):
        plt.plot( x_data[k], y_data[k], 'b-' )
    plt.show()

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
    t0 = time()
    print "Testing if P(x|mu,eta,nlc,s) runs"
    res = coupa_scene.P_of_x_given_measurements_nonlinear_class_speed(  x, 0 , 1.5 )
    print "result = %f" % res
    print "CPU time = %f \n" % (time()-t0)


    print "Testing if P(xT | linear, mu, eta ) runs"
    T = np.random.rand()
    xT = coupa_scene.mu + T*coupa_scene.eta
    t0 = time()
    res = coupa_scene.P_of_future_position_given_linear_class_and_measurements(xT, T )
    print "result = %f" % res
    print "CPU time = %f \n" % (time()-t0)


    print "Testing P( linear | mu,eta) runs"
    coupa_scene.set_mu( [0.5, -0.5])
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
    coupa_scene.eta = coupa_scene.director_field(0, 0.0, 0.6 )
    t0 = time()
    res = coupa_scene.P_of_nonlinear_class_and_speed_given_measurements( 0, 1.0 )
    print "result = "
    print res
    print "CPU time = %f \n" % (time()-t0)
