def get_trajectories( fname ):
    #a function to load trajectories from annotation data.
    # LOAD DATA
    import pandas as pd

    data = pd.read_csv( fname, sep=" ", usecols=[0,1,2,3,4,9])
    data.columns = ['index','x1','y1','x2','y2','label']
    # GET POSITIONS
    x=[]
    y=[]
    pedestrian_indices = set( data[ data['label']=='Pedestrian' ]['index'])
    import numpy as np

    #For the deathCircle videos the resolution is 1630 x 1948
    Nx = 1630
    Ny = 1948
    normalize = lambda x,N: 2*x / float(N) - 1.0

    for pid in pedestrian_indices:
        x1 = np.array(data[data['index']==pid]['x1'])
        x2 = np.array(data[data['index']==pid]['x2'])
        x.append( normalize( 0.5*(x1+x2) , Nx ) )
        y1 = np.array(data[data['index']==pid]['y1'])
        y2 = np.array(data[data['index']==pid]['y2'])
        y.append( -normalize( 0.5*(y1+y2) , Ny ) )
    return [x,y]


def smooth_trajectories( x_raw , y_raw ):
    #Smooths and extracts positions, velocities, and accelerations
    x_list = []
    y_list = []
    ax_list = []
    ay_list = []
    vx_list = []
    vy_list = []
    N_points = 0
    for x_traj,y_traj in zip(x_raw,y_raw):
        from scipy import signal
        sigma =5.0
        G = signal.gaussian(5*sigma,sigma)
        G = G / G.sum()
        x_smooth = signal.fftconvolve( x_traj, G , mode='valid' )
        y_smooth = signal.fftconvolve( y_traj, G , mode='valid' )
        N = x_smooth.size
        x_list.append( x_smooth[1:(N-1)] )
        y_list.append( y_smooth[1:(N-1)] )
        vx_list.append( 0.5*(-x_smooth[0:(N-2)] + x_smooth[2:N]) )
        vy_list.append( 0.5*(-y_smooth[0:(N-2)] + y_smooth[2:N]) )
        ax_list.append( -x_smooth[0:(N-2)] + 2*x_smooth[1:(N-1)]-x_smooth[2:N] )
        ay_list.append( -y_smooth[0:(N-2)] + 2*y_smooth[1:(N-1)]-y_smooth[2:N] )
    return x_list,vx_list,ax_list,y_list,vy_list,ay_list

import numpy as np
def eval_Hermites(x,deg):
    #computes the Hermite polynomials 0,...,deg at an array of points x.
    #returns a matrix of size (len(x),deg)
    H = np.zeros(deg+2)
    H[0] = 1.0
    H[1] = 2*x
    for n in range(2,deg+2):
        H[n] = 2*x*H[n-1]-2*(n-1)*H[n-2]
    return H

def EL_functional(x,v,a, deg_x = 20 , deg_v = 3):
    #computes the functional \ell^x(x,v,a)
    #only works if x,v,a are a single point.
    #you should be able to vector-rize this using parallel for-loops.
    partial_x = np.diag( 2*(np.arange(deg_x+1)+1),k=-1 )
    partial_v = np.diag( 2*(np.arange(deg_v+1)+1),k=-1 )
    h_of_x = eval_Hermites(x,deg_x)
    h_prime_of_x = partial_x.dot(h_of_x)
    h_of_v = eval_Hermites(v,deg_v)
    h_prime_of_v = partial_v.dot(h_of_v)
    h_double_prime_of_v = partial_v.dot(h_prime_of_v)
    t1 = np.kron( h_prime_of_x , h_prime_of_v*v )
    t2 = np.kron( h_of_x , h_double_prime_of_v*a )
    t3 = - np.kron( h_prime_of_x , h_of_v )
    return t1+t2+t3

def EL_functional_2d(x,vx,ax,y,vy,ay,deg_x=20,deg_v=3):
    h_of_x = eval_Hermites(x,deg_x)
    h_of_y = eval_Hermites(y,deg_x)
    h_of_vx = eval_Hermites(vx,deg_v)
    h_of_vy = eval_Hermites(vy,deg_v)
    e1 = reduce( np.kron, [EL_functional(x,vx,ax) , h_of_y , h_of_vy])
    e2 = reduce( np.kron, [h_of_x , h_of_vx , EL_functional(y,vy,ay)])
    return [e1,e2]

def display_trajectories(x_list,y_list,scene):
    from matplotlib import pyplot as plt
    directory_name = "../annotations/" + scene + "/video0/"
    im = plt.imread(directory_name+"reference.png")
    implot = plt.imshow(im, extent=[-1,1,-1,1])
    for x,y in zip(x_list,y_list):
        plt.plot(x,y,'b-')
    plt.show()
    return 0

if __name__ == '__main__':
    print "Let us test this baby"
    directory_name = "../annotations/deathCircle/video0/"
    file_name = "annotations.txt"
    fname = directory_name + file_name
    x_raw,y_raw = get_trajectories( fname )
    x_list,vx_list,ax_list,y_list,vy_list,ay_list = smooth_trajectories(x_raw, y_raw)
    display_trajectories( x_list,y_list , "deathCircle" )
    functionals = []
    for traj_tuple in zip(x_list,vx_list,ax_list,y_list,vy_list,ay_list):
        for arg in zip( *traj_tuple):
            functionals += EL_functional_2d(*arg)
    functionals_npy = np.stack( functionals )
    
    from time import time
    t0 = time()
    Q = np.einsum('ki,kj->ij', functionals_npy , functionals_npy )
    total_time = time() - t0
    print "Computing Q from ells tooks %f seconds" % total_time
    np.save( "Q_matrix")

