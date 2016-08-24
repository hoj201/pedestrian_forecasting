"""
pedestrian potentials is a suite of functions for learning potential functions 
"""

import numpy as np

def learn_potential( x_arr, y_arr , width, height, k_max = 6 ):
    """Returns the legendre coefficients of a potential function learned from a list of point in a 2D domain.

    args:
    x_arr -- iterable
    y_arr -- iterable (len(x_arr)==len(y_arr))
    domain -- iterable (dtype = int, len(domain) = 4)
    
    kwargs:
    None
    """

    # COST FUNCTION 
    V_scale = ( 1.2*(width/2) , 1.2*(height/2) )
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
        I = width*height*np.exp( - legval2d( x_grid/V_scale[0], y_grid/V_scale[1] , theta)).sum() / (res**2)
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


def prune_trajectories( curves ):
    """Given a list of curves, prune_trajectories(curves) returns a sublist of curves where the outlying curves (in terms of length) are removed

    args:
    curves -- list of numpy.ndarray

    kwargs:
    None
    """

    #Compute IQR
    log_length = lambda c: np.log( c.shape[1] )
    log_curve_lengths = map( log_length , curves )
    log_curve_lengths.sort()
    num_curves = len(curves)
    IQ_1 = log_curve_lengths[ num_curves / 4 ]
    IQ_3 = log_curve_lengths[ 3*num_curves / 4 ]
    IQR = IQ_3 - IQ_1

    #Determin which curves fall within IQR+/- 1.5*width range
    is_not_outlier = lambda c : log_length(c) > IQ_1 - 1.5*IQR or log_length(c) < IQ_3 + 1.5*IQR
    return filter( is_not_outlier , curves )


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
    return out

if __name__ == "__main__":
    print "Gathering trajectories"
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
    curves = map( np.vstack, zip(x_data, y_data) )
    curves = prune_trajectories( curves )
    clusters = cluster_trajectories( curves )

    print "Learning a potential for the first cluster"
    points_0 = np.hstack( clusters[1] )
    theta = learn_potential( points_0[0], points_0[1], width, height) 

    print "Plotting"
    from matplotlib import pyplot as plt
    x_grid,y_grid = np.meshgrid( np.linspace(-width/2,width/2,50) , np.linspace(-height/2,height/2,50))
    from numpy.polynomial.legendre import legval2d
    V_scale = ( 1.2*width/2, 1.2*height/2)
    V = legval2d(x_grid/V_scale[0] , y_grid/V_scale[1], theta)
    fig = plt.figure(figsize=(15,7))
    plt.contourf(x_grid,y_grid,V , 50, cmap='viridis')
    plt.colorbar()
    plt.plot( points_0[0] , points_0[1],'k*')
    plt.show()
