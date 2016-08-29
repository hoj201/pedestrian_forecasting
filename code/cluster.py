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
