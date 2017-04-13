import numpy as np
def _smart_LCSS(a, b, epsilon, delta):
    """
    takes
    F1: np.array([N, 2])
    F2: np.array([N, 2])
    epsilon: float
    delta: int
    """
    norm = lambda x: np.sqrt(np.dot(x,x))
    lengths = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if norm(x - y) < epsilon and np.abs(i-j) < delta:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
    sm = lambda x, y : x + y
    return max(reduce(sm, lengths))

def _naive_LCSS(F1, F2, epsilon, delta):
    """
    takes
    F1: np.array([N, 2])
    F2: np.array([N, 2])
    epsilon: float
    delta: int
    """

    T1 = len(F1)
    T2 = len(F2)

    
    if T1 == 0 and T2 == 0:
        return 0
    elif norm(F1[-1] - F2[-1]) < epsilon and np.abs(T1-T2) < delta:
        return 1 + _naive_LCSS(F1[:-1], F2[:-1], epsilon, delta)
    else:
        return max(_naive_LCSS(F1[:-1], F2, epsilon, delta), _naive_LCSS(F1, F1[:-1], epsilon, delta))

def LCSS(F1, F2, epsilon, delta):
    F1 = F1.transpose()
    F2 = F2.transpose()
    return 1 - float(_smart_LCSS(F1, F2, epsilon, delta))/min(len(F1), len(F2))

def cluster_trajectories_2(curves, epsilon, delta = 5):
    """
    takes:
    list(np.array([2, N]))
    """
    n_curves = len(curves)
    from sklearn.cluster import AffinityPropagation
    clusterer = AffinityPropagation(affinity='precomputed', convergence_iter=100)
    aff = np.zeros((n_curves, n_curves), dtype=float)
    for i in range(n_curves):
        print "{}% Done".format(i/float(n_curves) * 100)
        for j in range(i+1,n_curves):
            aff[i,j] = LCSS(curves[i],curves[j], epsilon, delta)
            aff[j,i] = aff[i,j]

    #clusterer.Affinity = aff
    cluster_labels = clusterer.fit_predict(aff)
    out = []
    for label in set(cluster_labels):
        cluster = map(lambda k: curves[k], filter(lambda k: cluster_labels[k]==label, range(n_curves)))
        out.append(cluster)
    return map( align_cluster, out)




def align_cluster(cluster):
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


def cluster_trajectories(curves):
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
    aff = np.zeros((n_curves, n_curves), dtype=float)
    for i in range(n_curves):
        for j in range(i+1,n_curves):
            aff[i,j] = np.exp(-distance_metric( X_2B_clstrd[i], X_2B_clstrd[j])**2)
            aff[j,i] = aff[i,j]

    #clusterer.Affinity = aff
    cluster_labels = clusterer.fit_predict(aff)
    out = []
    for label in set(cluster_labels):
        cluster = map(lambda k: curves[k], filter(lambda k: cluster_labels[k]==label, range(n_curves)))
        out.append(cluster)
    return map( align_cluster, out)

def prune_cluster(cluster):
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
        if len(cl) < 3 or portion[k] < 0.03:
            n_discarded += len(cl)
        else:
            new_clusters.append( cl )
    return n_discarded, new_clusters

def learn_potential(cluster, width, height, k_max=8, stride=30):
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
        V_mean = legval2d( 2*x_arr/width, 2*y_arr/height, theta).mean()
        k_span = np.arange( k_max+1)
        res = 2*(k_max+10)
        x_span = np.linspace(-width/2, width/2, res)
        y_span = np.linspace(-height/2, height/2, res)
        x_grid,y_grid = np.meshgrid(x_span, y_span)
        Area = width*height
        #TODO: Consider using scipy.dblquad to compute I
        I = Area*np.exp( - legval2d( 2*x_grid/width, 2*y_grid/height , theta)).mean()
        regularization = np.sqrt( np.einsum( 'ij,i,j', theta**2 , k_span**2 , k_span**2 ) )
        lambda_0 = 0.01
        return V_mean + np.log(I) + lambda_0 * regularization

    # CONSTRAINTS
    def potential_constraint(theta_flat ):
        return theta_flat[0]
    constraint_list = []
    constraint_list.append({'type':'eq', 'fun':potential_constraint })
    initial_guess = np.zeros( (k_max+1)**2 )

    # CALLBACK FUNCTIONS
    def cb_function(theta_flat):
        cb_function.bar.next()
        return 0
    import progress.bar
    cb_function.bar = progress.bar.Bar('Iteration', max=1000 )

    # MINIMIZE COST
    from scipy.optimize import minimize
    res = minimize( cost_function, initial_guess, constraints=constraint_list, callback = cb_function, options={'maxiter':cb_function.bar.max})

    # RETURN RESULT
    print res.message
    assert(res.success)
    return res.x.reshape( (k_max+1, k_max+1) )

def get_classes(curves, width, height, k_max=4):
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
    clusters = cluster_trajectories_2(curves, width/10)
    
    #Prune the clusters and keep track of how many agents you discard
    n_discarded = 0
    for k in range(len(clusters)):
        n, clusters[k] = prune_cluster(clusters[k])
        n_discarded +=n
    n, clusters = merge_small_clusters(clusters)
    n_discarded += n

    #print "There are %d nonlinear clusters" % len(clusters)
    #from matplotlib import pyplot as plt
    #fig, ax_arr = plt.subplots( len(clusters) )
    #for k,cl in enumerate(clusters):
    #    for curve in cl:
    #        ax_arr[k].plot( curve[0], curve[1] )
    #plt.show()

    #Compute P_of_c
    P_of_c = np.ones( len(clusters) + 1 )
    n_agents = n_discarded + reduce( lambda x,y: x+y, map( len , clusters ) )
    P_of_c[-1] = n_discarded / float(n_agents )
    P_of_c[:len(clusters)] = map( lambda c: len(c)/float(n_agents) , clusters)
    
    #Resetting P_of_c to a uniform distirbution
    #P_of_c /= P_of_c.size

    #Compute alphas
    alpha = np.zeros( (len(clusters)+1, k_max+1, k_max+1 ))
    for k in range(len(clusters)):
        alpha[k] = learn_potential( clusters[k] , width, height, k_max=k_max)
    return alpha, P_of_c, clusters

if __name__ == "__main__":
    print "Testing clustering routine"
    import numpy as np
    import process_data
    folder = '../annotations/bookstore/video3/'
    BB_ts_list, width, height = process_data.get_BB_ts_list(folder, label="Biker")
    curve_list = map(process_data.BB_ts_to_curve, BB_ts_list)

    import matplotlib.pyplot as plt
    for curve in curve_list:
        plt.plot(curve[0], curve[1],'b-')
    plt.grid()
    plt.axis('equal')
    plt.show()

    alpha, P_of_c, clusters = get_classes(curve_list, width, height)
    n_cluster = len(clusters)
    print "n_cluster = %d \n" % n_cluster
    fig, ax_arr = plt.subplots( n_cluster , 1 , figsize = (5,10))
    for k,cl in enumerate(clusters):
        for curve in cl:
            ax_arr[k].plot( curve[0] , curve[1] , 'b-')
            ax_arr[k].axis( [-width/2, width/2, -height/2, height/2])
    plt.title("Clusters")
    plt.show()

    print "Testing computation of prior"
    print "P(c) = " 
    print P_of_c
    print "Sum = %f" % P_of_c.sum()
