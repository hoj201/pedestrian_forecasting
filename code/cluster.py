import numpy as np

def align_cluster( cluster ):
    """ returns cluster with curves aligned

    args:
        cluster (list(numpy.ndarray) ): each element is a numpy array of shape (2,N)

    returns:
        cluster (list(numpy.ndarray) ): each element has the same shape as the original cluster
    """
    x_0 = cluster[0][:,0]
    x_f = cluster[0][:,-1]
    distance = lambda x : np.dot(x,x)
    reverse_or_not = lambda x: x[:,::-1] if distance(x[:,0]-x_0) > distance(x[:,0]-x_f) else x
    return [ reverse_or_not(c)  for c in cluster ]


def cluster_trajectories( curves ):
    """Given a list of curves, cluster_trajectories will cluster them."""
    n_curves = len(curves)
    X_2B_clstrd = np.zeros( (n_curves, 4) )
    X_2B_clstrd[:,0] = np.array( [ curves[k][0, 0] for k in range(n_curves) ] )
    X_2B_clstrd[:,1] = np.array( [ curves[k][1, 0] for k in range(n_curves) ] )
    X_2B_clstrd[:,2] = np.array( [ curves[k][0,-1] for k in range(n_curves) ] )
    X_2B_clstrd[:,3] = np.array( [ curves[k][1,-1] for k in range(n_curves) ] )
        
    for col in range( 4 ):
        X_2B_clstrd[:,col] /=  X_2B_clstrd[:,col].std()
        
    def distance_metric(a,b):
        #A distance metric on R^4 modulo the involution
        #(x0,x2,x3,x4) -> (x3,x4,x1,x2)
        d = lambda a,b : np.sqrt( np.sum( (a-b)**2 ) )
        T = lambda x: np.array([x[2],x[3],x[0],x[1]])
        return min( d(a,b) , d(T(a),b) )
    from sklearn.cluster import AffinityPropagation
    clusterer = AffinityPropagation(affinity='precomputed', convergence_iter=100)
    aff = np.zeros((n_curves, n_curves))
    for i in range(n_curves):
        for j in range(i+1,n_curves):
            aff[i,j] = np.exp(-distance_metric( X_2B_clstrd[i], X_2B_clstrd[j])**2)
            aff[j,i] = aff[i,j]

    #clusterer.Affinity = aff
    cluster_labels = clusterer.fit_predict(aff)
    out = []
    for label in set( cluster_labels):
        cluster = map( lambda k: curves[k] , filter( lambda k: cluster_labels[k] == label , range( n_curves) ) )
        out.append( cluster )
    return map( align_cluster, out)


def prune_cluster( cluster ):
    """ Removes the abnormally long/short trajectories from a cluster

    args:
        cluster (list(numpy.ndarray) ) : clusters of trajectories

    returns:
        cluster (list(numpy.ndarray) ) : the pruned clusters
    """
    def length_of_traj( traj ):
        n = traj.shape[1]
        u = traj[0,1:] - traj[0,:n-1]
        v = traj[1,1:] - traj[1,:n-1]
        return np.sqrt(u**2+v**2).sum()
    n = len(cluster)
    lengths = map( length_of_traj , cluster)
    lengths.sort()
    #Compute IQR
    Q1 = lengths[n/4]
    Q3 = lengths[3*n/4]
    IQR = Q3-Q1

    #Compute which to discard and count hoe many agents you discard
    keep_it = lambda traj: length_of_traj( traj ) < Q3+1.5*IQR and length_of_traj( traj ) > Q1-1.5*IQR
    bool2int = lambda b: 1 if b else 0
    n_keep = reduce( lambda x,y:x+y, map( bool2int, map( keep_it, cluster) ) )
    n_discarded = len(cluster) - n_keep
    
    return n_discarded, filter( keep_it , cluster )


def merge_small_clusters( clusters):
    """ returns clusters where each has a minimum size and one cluster is just a linear predictor

    args:
        clusters ( list(list(numpy.ndarray)) ): each element is a numpy array of shape (2,N)

    returns:
        clusters (list(list(numpy.ndarray)) + [Null_ls]*N_0:
    """
    new_clusters = []
    n_discarded = 0
    N_curves = reduce( lambda x,y: x+y, map( len, clusters ) )
    portion = [ len(c) / float(N_curves) for c in clusters]
    for k,cl in enumerate(clusters):
        if len(cl) < 3 or portion[k] < 0.1:
            n_discarded += len(cl)
        else:
            new_clusters.append( cl )
    return n_discarded, new_clusters


def learn_potential( cluster , V_scale, k_max = 4, stride=30):
    """Returns the legendre coefficients of a potential function learned from a list of point in a 2D domain.

    args:
        cluster (list(numpy.ndarray): list of trajectories.

    kwargs:
        stride (int): stride length to use along each trajectory.  Default=10

    returns:
        out (numpy.ndarray) : coeffs for 2D Legendre series describing potential funciton. shape = (k_max+1, k_max+1)
    """
    xy_arr = np.hstack( cluster )
    x_arr = xy_arr[0][::stride]
    y_arr = xy_arr[1][::stride]

    # COST FUNCTION 
    def cost_function(theta_flat ):
        V_sum = 0
        N = 0
        theta = theta_flat.reshape( (k_max+1, k_max+1))
        from numpy.polynomial.legendre import legval2d,leggauss
        V_mean = legval2d( x_arr/V_scale[0], y_arr/V_scale[1], theta).mean()
        k_span = np.arange( k_max+1)
        res = 2*(k_max+10)
        x_span = np.linspace(-V_scale[0], V_scale[0], res)
        y_span = np.linspace(-V_scale[1], V_scale[1], res)
        x_grid,y_grid = np.meshgrid(x_span, y_span)
        Area = V_scale[0]*V_scale[1]*4
        #TODO: Consider using scipy.dblquad to compute I
        I = Area*np.exp( - legval2d( x_grid/V_scale[0], y_grid/V_scale[1] , theta)).mean()
        regularization = np.sqrt( np.einsum( 'ij,i,j', theta**2 , k_span**2 , k_span**2 ) )
        lambda_0 = 1e-4
        return V_mean + np.log(I) + lambda_0 * regularization

    # CONSTRAINTS
    def potential_constraint(theta_flat ):
        return theta_flat[0]
    constraint_list = []
    constraint_list.append({'type':'eq', 'fun':potential_constraint })
    initial_guess = np.zeros( (k_max+1)**2 )

    # CALLBACK FUNCTIONS
    def cb_function( theta_flat ):
        global k_max,curves
        cb_function.iteration += 1
        from progress_bar import update_progress
        update_progress( cb_function.iteration / float(cb_function.max_iteration) )
        return 0
    cb_function.iteration = 0
    cb_function.max_iteration = 1000

    # MINIMIZE COST
    from scipy.optimize import minimize
    res = minimize( cost_function, initial_guess, constraints=constraint_list, callback = cb_function, options={'maxiter':cb_function.max_iteration})

    # RETURN RESULT
    print res.message
    assert(res.success)
    return res.x.reshape( (k_max+1, k_max+1) )


def get_classes( curves, V_scale, k_max=4 ):
    """ Given curves returns coefficients and probabilities and clusters

    args:
        clusters ( list(numpy.ndarray) ): a list of trajectories
        V_scale : tuple of length 2.  Defines the size of the domain

    kwargs:
        k_max : degree of max polynomial to use in the negative likelihood field.

    returns:
        alpha, P_of_c, clusters_pruned


    NOTE: alpha[k] and clusters_pruned[k] have probabilitiy P_of_c[k].  k=-1 corresponds to a linear predictor
    """
    #TODO: Test this.  Note, alpha[k] corresponds to P_of_c[k].  alpha[-1] corresponds to the linear predictor.
    clusters = cluster_trajectories( curves )
    
    #Prune the clusters and keep track of how many agents you discard
    n_discarded = 0
    for k in range(len(clusters)):
        n, clusters[k] = prune_cluster( clusters[k] )
        n_discarded +=n
    n, clusters = merge_small_clusters( clusters )
    n_discarded += n

    #print "There are %d nonlinear clusters" % len(clusters)
    #from matplotlib import pyplot as plt
    #fig, ax_arr = plt.subplots( len(clusters) )
    #for k,cl in enumerate(clusters):
    #    for curve in cl:
    #        ax_arr[k].plot( curve[0], curve[1] )
    #plt.show()

    #Compute P_of_c
    P_of_c = np.zeros( len(clusters) + 1 )
    n_agents = n_discarded + reduce( lambda x,y: x+y, map( len , clusters ) )
    P_of_c[0] = n_discarded / float(n_agents )
    P_of_c[1:] = map( lambda c: len(c)/float(n_agents) , clusters)

    #Compute alphas
    alpha = np.zeros( (len(clusters)+1, k_max+1, k_max+1 ))
    for k in range(len(clusters)):
        alpha[k] = learn_potential( clusters[k] , V_scale, k_max=k_max)
    return alpha, P_of_c, clusters

if __name__ == "__main__":
    print "Testing clustering routine"
    import numpy as np
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
    import matplotlib.pyplot as plt
    for k in range(len(x_data)):
        plt.plot(x_data[k], y_data[k],'b-')
    plt.grid()
    plt.axis('equal')
    plt.show()

    curves = map( np.vstack , zip(x_data, y_data) )
    clusters = cluster_trajectories( curves )
    n_cluster = len(clusters)
    fig, ax_arr = plt.subplots( n_cluster , 1 , figsize = (5,10))
    for k,cl in enumerate(clusters):
        for curve in cl:
            ax_arr[k].plot( curve[0] , curve[1] , 'b-')
            ax_arr[k].axis( [-width/2, width/2, -height/2, height/2])
    plt.title("Clusters")
    plt.show()

    print "Testing cluster merging routine"
    new_clusters = merge_small_clusters( clusters )
    n_cluster = len(new_clusters)
    fig, ax_arr = plt.subplots( n_cluster-1 , 1 , figsize = (5,10))
    for k,cl in enumerate(new_clusters[1:]):
        for curve in cl:
            ax_arr[k].plot( curve[0] , curve[1] , 'b-')
            ax_arr[k].axis( [-width/2, width/2, -height/2, height/2])
    plt.show()
    plt.title("Just the large clusters")


    print "Testing computation of prior"
    P_of_c = compute_prior( new_clusters )
    print "P(c) = " 
    print P_of_c
    print "Sum = %f" % P_of_c.sum()

