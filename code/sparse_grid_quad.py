import numpy as np
from pandas import read_csv
sparse_grid_nodes_3d = read_csv('./sparse_grid_data/points_3d.dat', sep=',', header=None).values
sparse_grid_weights_3d = read_csv('./sparse_grid_data/weights_3d.dat', sep=',', header=None).values.flatten()
sparse_grid_nodes_2d = read_csv('./sparse_grid_data/points_2d.dat', sep=',', header=None).values
sparse_grid_weights_2d = read_csv('./sparse_grid_data/weights_2d.dat', sep=',', header=None).values.flatten()


def sparse_grid_quad_2d( integrand, x_min, x_max, y_min, y_max ):
    #Computes the sparse grid quadrature of the integrand over the box [x_min,x_max] x [y_min, y_max]
    transformed_nodes = np.zeros_like( sparse_grid_nodes_2d )
    transformed_nodes[:,0] = sparse_grid_nodes_2d[:,0]*(x_max-x_min) + x_min
    transformed_nodes[:,1] = sparse_grid_nodes_2d[:,1]*(y_max-y_min) + y_min
    vals_on_nodes = integrand( transformed_nodes.transpose() )
    Vol = (x_max - x_min ) * (y_max - y_min )
    return np.dot( vals_on_nodes , sparse_grid_weights_2d ) * Vol

def sparse_grid_quad_3d( integrand, x_min, x_max, y_min, y_max, z_min, z_max ):
    #Computes the sparse grid quadrature of the integrand over the box [x_min,x_max] x [y_min, y_max]
    transformed_nodes = np.zeros_like( sparse_grid_nodes_3d )
    transformed_nodes[:,0] = sparse_grid_nodes_3d[:,0]*(x_max-x_min) + x_min
    transformed_nodes[:,1] = sparse_grid_nodes_3d[:,1]*(y_max-y_min) + y_min
    transformed_nodes[:,2] = sparse_grid_nodes_3d[:,2]*(z_max-z_min) + z_min
    vals_on_nodes = integrand( transformed_nodes.transpose() )
    Vol = ( x_max-x_min ) * ( y_max - y_min) * (z_max - z_min )
    return np.dot( vals_on_nodes , sparse_grid_weights_3d ) * Vol
