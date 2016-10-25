import numpy as np
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    import process_data
    folder = '../annotations/coupa/video2/'
    x_data, y_data, width, height= process_data.get_trajectories(folder,label="Biker")

    curve_ls = [ np.vstack([x,y]) for (x,y) in zip( x_data, y_data ) ]
    from sklearn.cross_validation import train_test_split
    train_set, test_set = train_test_split( curve_ls, random_state = 0 )
    from scene import scene
    coupa_scene = scene( train_set, V_scale )
    coupa_scene.set_mu( np.zeros(2) )
    coupa_scene.set_eta( np.ones(2) )
    print "mu = "
    print coupa_scene.mu
    print "eta = "
    print coupa_scene.eta
    
    out = visualize_cluster( coupa_scene, 0 )

