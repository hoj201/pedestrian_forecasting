import numpy as np
import matplotlib.pyplot as plt
from scene import Scene

def visualize_cluster( scene, k ):
    """ Displays a plot of all the clusters and there potential functions

    args:
        scene:
        k: (int)

    returns:
        plot of requested cluster, as well as a potential function and a vector field.
    """
    assert( k < scene.num_nl_classes )
    cluster = scene.clusters[k]
    fig, ax_arr = plt.subplots( 1 , 2 , figsize=(15,5) )
    width = scene.width
    height = scene.height
    alpha = scene.alpha_arr[k]
    from numpy.polynomial.legendre import legval2d
    X_grid, Y_grid = np.meshgrid( np.linspace( -width/2, width/2, 20),
            np.linspace( -height/2, height/2, 20)
            )
    V = legval2d( 2*X_grid / width, 2*Y_grid / height, alpha )
    V -= V.min()
    p_of_x = np.exp( - V)
    ax_arr[0].contourf( X_grid, Y_grid, p_of_x, 40, cmap = 'plasma',
            interpolation='cubic')
    for xy in cluster:
        ax_arr[0].plot( xy[0] , xy[1] , 'w-' )
    ax_arr[0].axis('equal')
    UV = scene.director_field_vectorized( k, np.vstack([X_grid.flatten(), Y_grid.flatten() ] ) )
    U_grid = UV[0].reshape( X_grid.shape)
    V_grid = UV[1].reshape( X_grid.shape)
    ax_arr[1].quiver( X_grid, Y_grid, U_grid, V_grid, scale = 30)
    ax_arr[1].axis('equal')
    plt.show()
    return -1


def singular_distribution_to_image(pts, weights, domain, res=(50,50)):
    """ Converts a singular distribution into an image for plotting.

    args:
        pts: np.array.shape = (2,N)
        weights: np.array.shape = (N,)
        domain: (x_min, x_max, y_min, y_max)
        
    kwargs:
        res: (uint, uint)

    returns:
        np.array.shape = (res[0], res[1])
    """
    #Sort the points and weights by the x-component of each point
    indices = np.argsort(pts[0])
    pts = pts[:,indices]
    weights = weights[indices]

    #Partition space
    partition_x = np.linspace(domain[0], domain[1], res[0]+1)
    partition_y = np.linspace(domain[2], domain[3], res[1]+1)

    #Initialize output array
    im = np.zeros(res)

    #For each box of the partition, find the points in the box
    for i in range(res[0]):
        lbx = partition_x[i]
        ubx = partition_x[i+1]
        start = pts[0].searchsorted(lbx)
        end = pts[0].searchsorted(ubx)
        pts_x = pts[:,start:end]
        weights_x = weights[start:end]

        #Sort with respect to y-component
        indices = np.argsort(pts_x[1])
        pts_x = pts_x[:,indices]
        weights_x = weights_x[indices]
        for j in range(res[1]):
            lby = partition_y[j]
            uby = partition_y[j+1]
            start = pts_x[1].searchsorted(lby)
            end = pts_x[1].searchsorted(uby)
            im[i,j] = weights_x[start:end].sum()
    return im

if __name__ == '__main__':
    with open("test_scene.pkl", "rb") as f:
        import pickle
        scene = pickle.load(f)
    visualize_cluster( scene, 0)

    print "Testing routine for visualizing singular distributions"
    weights = np.ones(100000)
    pts = np.random.randn(2, weights.size)
    domain = (-2,2,-2,2)

    im = singular_distribution_to_image( pts, weights, domain)
    plt.imshow(im)
    plt.show()

