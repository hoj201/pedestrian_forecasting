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


def compute_prior( clusters ):
    """ returns probability function over the clusters

    args:
        clusters ( list(list(numpy.ndarray)) ): each element is a numpy array of shape (2,N)

    returns:
        P_of_c numpy.ndarray: A 1D-array of positive floats which sum to 1.0
    """
    N_curves = reduce( lambda x,y: x+y, map( len, clusters ) )
    P_of_c = [ len(c) / float(N_curves) for c in clusters]
    return np.array( P_of_c )


def merge_small_clusters( clusters):
    """ returns clusters where each has a minimum size and one cluster is just a linear predictor

    args:
        clusters ( list(list(numpy.ndarray)) ): each element is a numpy array of shape (2,N)

    returns:
        clusters (list(list(numpy.ndarray)) + [Null_ls]*N_0:
    """
    new_clusters = [["Linear",] ,]
    N_linear = 0
    P_of_c = compute_prior( clusters )
    for cl in clusters:
        if len(cl) < 5 or P_of_c[k] < 0.1:
            N_linear += len(cl)
        else:
            new_clusters.append( cl )
    new_clusters[0] = ["Linear"]*N_linear
    return new_clusters


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

