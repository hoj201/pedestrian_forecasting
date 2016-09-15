import numpy as np

def modified_Hausdorff_distance( A, B ):
    """ Returns the modified Hausdorff distance between two point sets in R2

    args:
        A (numpy.array): shape = (N_A,2)
        B (numpy.array): shape = (N_B,2)

    returns:
        d (float)
    """
    from scipy.spatial.distance import cdist
    d_matrix = cdist( A , B )
    out1 = np.mean( np.min( d_matrix, axis = 1) )
    out2 = np.mean( np.min( d_matrix, axis = 0) )
    return max( [out1, out2] )


if __name__ == '__main__':
    a = np.random.randn( 10,2)
    b = np.random.randn( 100,2) + np.array( [ 1.0 , 0.0 ] )
    print modified_Hausdorff_distance( a,b)
