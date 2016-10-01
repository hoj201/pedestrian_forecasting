import numpy as np
from sparse_grid_quad import sparse_grid_quad_2d, sparse_grid_quad_3d

sigma_x = 0.05
sigma_v = 2*sigma_x

class Scene():
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
        print "Top speed = %f" % self.s_max

        #Learn the  agent_classes
        from cluster import get_classes
        alpha_arr, P_of_c, clusters = get_classes( curve_ls, V_scale )
        self.alpha_arr = alpha_arr
        self.P_of_c = P_of_c
        print P_of_c
        self.V_scale = V_scale 
        self.clusters = clusters

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

    # SETTERS
    def set_mu(self, mu ):
        self.mu = mu

    def set_eta(self, eta):
        self.eta = eta
    #--------------------------------------------------------------------------------
    # METHODS
    #--------------------------------------------------------------------------------


    #-------------------- VECTOR FIELD ROUTINES -------------------------------------

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

    #-------------------- PROBABILITY DENSITY PREDICTORS -----------------------
    def predict_pdf( self, X_grid, Y_grid, T, show_output = False):
        """ returns the values of a probability density function at time T at given positions

        args:
            X_grid:
            Y_grid:
            T
        returns:
            rho_T
        """
        N_steps = 100
        s_ls = np.linspace( -self.s_max, self.s_max, 2*N_steps-1 )
        ds = s_ls[1] - s_ls[0]
        rho_T = np.zeros( X_grid.size )
        x0,y0 = X_grid.flatten(), Y_grid.flatten()
        P_linear = 1.0
        helper = lambda x,y : self.P_of_x_given_mu( np.array([x,y]) )
        for k in range( self.num_nl_classes ):
            P_cs = [ self.P_of_nl_class_and_speed_given_measurements(k,s) for s in s_ls ]
            P_cs = np.array( P_cs )
            P_c = P_cs.sum() * ds
            P_linear -= P_c
            if show_output:
                print "P_c = %f" % P_c
            tol = 1e-3
            if P_c < tol:
                if show_output:
                    print "Skipping computation for class c_%d.  P(c_%d | mu )=%g < %g \n" % (k,k,P_c,tol)
                continue
            dynamics = lambda x,jac=False: self.director_field_vectorized( k, x, jac=jac)
            from particle_advect import advect_vectorized as advect
            x_t,y_t,w_t = advect( dynamics, x0, y0 , T*self.s_max, N_steps )
            rho_positive = helper( x_t, y_t ) * w_t
            x_t,y_t,w_t = advect( dynamics, x0, y0 , -T*self.s_max, N_steps )
            rho_negative = helper( x_t, y_t ) * w_t
            rho = np.concatenate( [ rho_negative, rho_positive[1:] ], axis=0 ) #we exclude one point so as not to double count t=0

            #ADVECT AND ADD TO OUT, WEIGHTED BY P(c,s|mu,eta)
            rho_T += np.dot( P_cs, rho ) * ds
            if show_output:
                print "c_%d case computed.\n" % k

        # ADD LINEAR TERM
        if show_output:
            print "P_linear = %f" % P_linear
        rho_T = rho_T.reshape( X_grid.shape )
        if P_linear > 1e-3:
            rho_linear = self.predict_pdf_linear( X_grid, Y_grid, T )
            rho_T += rho_linear*P_linear
        return rho_T

    def predict_pdf_linear( self, X_grid, Y_grid, T):
        """ returns the predicted pdf under the a Linear predictor.

        args:
            X_grid:
            Y_grid:
            T:

        returns:
            rho_linear:
        """
        x0, y0 = X_grid.flatten(), Y_grid.flatten()
        G = lambda x, sigma: np.exp( - x**2 / (2*sigma**2) ) / np.sqrt( 2*np.pi*sigma**2 )
        sigma = np.sqrt( sigma_x**2 + T**2 * sigma_v**2 )
        rho_linear = G( x0 - self.mu[0] - T*self.eta[0] , sigma)
        rho_linear *= G( y0 - self.mu[1] - T*self.eta[1] , sigma)
        return rho_linear.reshape( X_grid.shape)


    #-------------------- SINGLE TRAJECTORY PREDICTOR --------------------------
    def predict_trajectory(self, t_final, N_steps):
        """ returns a trajectory based upon the most likely class

        args:
            t_final: float, final time
            N_steps: int, number of time-steps to integrate

        returns:
            x_arr: np array, shape (2,N_steps)
        """
        # FIND MOST LIKELY CLASS.
        s_span = np.linspace( -self.s_max, self.s_max, 30 )
        from itertools import product
        k,s = max( product( range( self.num_nl_classes), s_span ), key = lambda x: self.P_of_nl_class_and_speed_given_measurements( *x ) )

        P_lin = self.P_of_linear_given_measurements()
        if P_lin > self.P_of_nl_class_and_speed_given_measurements(k,s):
            t_span = np.linspace( 0.0, t_final, N_steps )
            return self.mu + np.outer( t_span, self.eta )
        import hoj_odeint
        f = lambda x: self.director_field(k,x) * s
        return hoj_odeint.rk4( f, self.mu, t_final, N_steps )[0]



    #-------------------- POSTERIOR ROUTINES ----------------------------------------
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
        epsilon = np.sqrt(self.V_scale[0]*self.V_scale[1]) / 1e4
        eta_hat = self.eta / ( epsilon + np.sqrt( np.dot(self.eta, self.eta) ) )
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
        if np.abs(s) > self.s_max:
            return 0.0

        if not hasattr( self, 'Z_cs_given_meas' ):
            print "Computing normalizing constant for P_of_nl_class_and_speed_given_measurements"
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
            v = s*self.director_field( k, xy )
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
    folder = '../annotations/coupa/video2/'
    BB_ts_list, V_scale = process_data.get_BB_ts_list(folder,label="Biker")

    curve_ls = map( process_data.BB_ts_to_curve, BB_ts_list )
    from sklearn.cross_validation import train_test_split
    train_set, test_set = train_test_split( curve_ls, random_state = 0 )

    coupa_scene = Scene( train_set, V_scale )
    print "Display clusters"
    for k in range( coupa_scene.num_nl_classes ):
        from visualization_routines import visualize_cluster
        visualize_cluster( coupa_scene, k )
    
    coupa_scene.set_mu( np.zeros(2) )
    coupa_scene.set_eta( np.ones(2) )
    print "mu = "
    print coupa_scene.mu
    print "eta = "
    print coupa_scene.eta

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
    res = coupa_scene.P_of_nl_class_and_speed_given_measurements( 0, 0.0 )
    print "result = %f" % res
    print "CPU time = %f \n" % (time()-t0)
    res = coupa_scene.P_of_nl_class_and_speed_given_measurements( 0, 1.0 )
    print "result2 = %f" % res

    print "Testing if P(linear | mu,eta) + \sum_k \int P(ck,s|mu,eta)ds = 1"
    I = 0.
    s_min = -coupa_scene.s_max
    s_max = coupa_scene.s_max
    res = 200.0
    ds = (s_max - s_min ) / (res-1)
    for k in range( coupa_scene.num_nl_classes ):
        integrand = lambda s: coupa_scene.P_of_nl_class_and_speed_given_measurements( k, s)
        I += sum( [ integrand(s)*ds for s in np.linspace(s_min, s_max, res ) ] ) 
    I += coupa_scene.P_of_linear_given_measurements()
    print "Sum = %f\n" % I

    X_grid, Y_grid = np.meshgrid(
            np.linspace( -coupa_scene.V_scale[0] , coupa_scene.V_scale[0], 40),
            np.linspace( -coupa_scene.V_scale[1] , coupa_scene.V_scale[1], 40)
            )

    print "Testing prediction routines"
    rho_grid_0 = coupa_scene.P_of_x_given_mu( [X_grid, Y_grid] )
    t0 = time()
    rho_grid = coupa_scene.predict_pdf( X_grid, Y_grid, 30.0 )
    print "CPU time = %f \n" % (time() - t0 )
    from matplotlib import pyplot as plt
    fig, ax_arr = plt.subplots(2,1, figsize=(10,5))
    cs = ax_arr[0].contourf( X_grid, Y_grid, rho_grid_0, 50, cmap = 'viridis' )
    ax_arr[0].axis( 'equal' )

    cs = ax_arr[1].contourf( X_grid, Y_grid, rho_grid , 50, cmap='viridis')
    ax_arr[1].axis( 'equal' )
    plt.show()


    print "Testing prediction for a single class."
    t_final = 3.0
    N_steps = 100
    x_arr = coupa_scene.predict_trajectory( t_final, N_steps )
    #print x_arr
