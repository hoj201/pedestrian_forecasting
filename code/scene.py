import numpy as np
import pickle
import process_data

class Scene():
    """ A class which learns stuff from a list of trajectories.
    
    attributes:
        num_nl_classes: This is the number of nonlinear vector fields
        P_of_c: 1d numpy array of length = num_nl_classes +1.
        width: width of the domain
        height: height of the domain
        alpha_arr:  A coefficient array for the potential function V_k.
        theta_arr:  A coefficient array for the angle-fields
        s_max: The top observed speed
        bbox_width: float
        bbox_velocity_width: Same as bbox_width but in velocity space.
        sigma_L: standard deviation of velocity
        sigma_x: standard deviation of position measurements
        sigma_v: standard deviation of velocity measurements
        kappa: indicates how much to blur the output of generate_distributions.py
    
    methods:
       director_field_vectorized:  A routine for computing a vector field.
       ...  A bunch of shit which we wont use.

    Notes:
        1) V_k(x,y) = \sum_{ij} alpha_arr[k,i,j] L_i( x / width ) L_j( y / height )
        2) X_k(x,y) = ( cos(theta_k(x,y), \sin( \theta_k(x,y) ) )
           where theta = \sum_{ij} theta_arr[k,i,j] L_i( x / width ) L_j( y / height )
        3) P_of_c[-1] = probability of linear class
        4) Given a measurement (x,y), the associated bounding box is [x-width/2,x+width/2]x[y-width/2,y+width/2]
 

    """
    def __init__( self, BB_ts_ls, width, height, k_max_theta=15, k_max_vf=15 ):
        """ Initializer

        args:
            BB_ts_ls : list of BBox time series.
            width: float
            height: float

        kwargs:
            k_max_theta : int
            k_max_vf : int
        """
       
        #Learn the top speed
        curve_ls = map( process_data.BB_ts_to_curve, BB_ts_ls )
        def top_speed( curve ):
            fd_width = 4
            n = curve.shape[1]
            u = ( curve[:,fd_width:] - curve[:,:n-fd_width] ) / float(fd_width)
            return ( np.sqrt( u[0]**2 + u[1]**2 ) ).max()

        self.bbox_width = process_data.get_bbox_width( BB_ts_ls )
        self.bbox_velocity_width = process_data.get_bbox_velocity_width( BB_ts_ls )
        self.s_max = max( map( top_speed, curve_ls ) )
        self.sigma_L = process_data.get_std_velocity( BB_ts_ls )
        self.sigma_x = process_data.get_std_measurement_noise( curve_ls )
        self.sigma_v = 2*self.sigma_x

        #Learn the  agent_classes
        from cluster import get_classes
        alpha_arr, P_of_c, clusters = get_classes( curve_ls, width, height)
        self.alpha_arr = alpha_arr
        self.P_of_c = P_of_c
        print P_of_c
        self.width = width
        self.height = height
        self.clusters = clusters

        #Learn a vector field for each class
        from director_field import trajectories_to_director_field
        X_ls = [] 
        k_max_theta = 4 #max degree of polynomial for angle field
        self.k_max_theta = k_max_theta
        self.num_nl_classes = len(clusters)
        theta_coeffs = np.zeros(
                (self.num_nl_classes, self.k_max_theta+1, self.k_max_theta+1)
                )
        for k,cl in enumerate(clusters):
            theta_coeffs[k] = trajectories_to_director_field(
                    cl, width, height,
                    k_max=self.k_max_theta
                    )
        self.theta_coeffs = theta_coeffs

        #Learn yet another variance
        kappa_per_curve = np.zeros(len(curve_ls))
        kappa_per_class = np.zeros(self.num_nl_classes)
        for i, curve in enumerate(curve_ls):
            #For each training compute x0, avg_speed
            x0 = curve[:,0]
            n = curve.shape[1]
            dx = curve[:,1:] - curve[:,:n-1]
            speed = sum(np.sqrt(dx[0]**2+dx[1]**2)) / (n-1)
            t_span = np.arange(n)
            #for each k, generate a curve.
            for k in range(self.num_nl_classes):
                ode_forward = lambda x,t: speed * self.director_field(k,x)
                ode_backward = lambda x,t: -speed * self.director_field(k,x)
                from scipy.integrate import odeint
                forward_curve = odeint(ode_forward, x0, t_span).transpose()
                backward_curve = odeint(ode_backward, x0, t_span).transpose()
                #Compute variance of (generated-given)/t -> kappa_k
                forward_std = ((curve-forward_curve)/(t_span+1)).std()
                backward_std = ((curve-backward_curve)/(t_span+1)).std()
                kappa_per_class[k] = min( forward_std, backward_std )
            kappa_per_curve[i] = kappa_per_class.min()
        self.kappa = kappa_per_curve.mean()
        print "kappa = %f" % self.kappa

    #--------------------------------------------------------------------------
    # METHODS
    #--------------------------------------------------------------------------
    def director_field(self, k, x, jac=False ):
        """ Returns the value of the vector field for a nonlinear class.

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
                x[0]/self.width,
                x[1]/self.height,
                self.theta_coeffs[k]
                )
        #NOTE:  Yes, I know this scaling seems off by a factor of 2.  At the moment, this is correct. However, this should be refactored so that we use a scaling convention that is consistent with the rest of the code-base (e.g. posteriors.x_given_k
        out1 = np.array([np.cos(theta), np.sin(theta)])
        if jac:
            dtheta_dx = legval2d(
                        x[0]/self.width,
                        x[1]/self.height,
                        legder( self.theta_coeffs[k], axis=0)
                        ) / self.width
            dtheta_dy = legval2d(
                        x[0]/self.width,
                        x[1]/self.height,
                        legder( self.theta_coeffs[k], axis=1)
                        ) / self.height
            out2 = np.zeros( (2,2) )
            out2[0,0] = - out1[1]*dtheta_dx
            out2[0,1] = - out1[1]*dtheta_dy
            out2[1,0] = out1[0]*dtheta_dx
            out2[1,1] = out1[0]*dtheta_dy
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
                x[0]/self.width,
                x[1]/self.height,
                self.theta_coeffs[k]
                )
        out1 = np.array([np.cos(theta), np.sin(theta)])
        if jac:
            dtheta_dx = legval2d(
                        x[0]/self.width,
                        x[1]/self.height,
                        legder( self.theta_coeffs[k], axis=0)
                        ) / self.width
            dtheta_dy = legval2d(
                        x[0]/self.width,
                        x[1]/self.height,
                        legder( self.theta_coeffs[k], axis=1)
                        ) / self.height
            N = x.shape[1]
            out2 = np.zeros((2,2,N))
            out2[0,0,:] = -out1[1]*dtheta_dx
            out2[0,1,:] = -out1[1]*dtheta_dy
            out2[1,0,:] = out1[0]*dtheta_dx
            out2[1,1,:] = out1[0]*dtheta_dy
            return out1, out2
        return out1


if __name__ == "__main__":
    print "Testing initializer"
    
    folder = '../annotations/coupa/video2/'
    BB_ts_list, width, height = process_data.get_BB_ts_list(folder,label="Biker")

    #curve_ls = map( process_data.BB_ts_to_curve, BB_ts_list )
    from sklearn.cross_validation import train_test_split
    train_set, test_set = train_test_split( BB_ts_list, random_state = 0 )

    test_scene = Scene( train_set, width, height )
    print "Display clusters"
    for k in range( test_scene.num_nl_classes ):
        from visualization_routines import visualize_cluster
        visualize_cluster( test_scene, k )
   
    print "P(k) = "
    print test_scene.P_of_c

    print "\sum_k P(k) = {}".format( test_scene.P_of_c.sum())
    response = raw_input("Would you like to pickle this scene? y/n")
    if response == "y":
        with open("test_scene.pkl", "ws") as f:
            pickle.dump( test_scene, f)

        with open("test_set.pkl", "ws") as f:
            pickle.dump( test_set, f)
