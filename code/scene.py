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
        sigma_v: standard deviation of velocity
    
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
    def __init__( self, BB_ts_ls, width, height, k_max_theta=6, k_max_vf=10 ):
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
        self.sigma_v = process_data.get_std_velocity( BB_ts_ls )

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


