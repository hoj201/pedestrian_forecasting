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
    for pid in pedestrian_indices:
        x1 = np.array(data[data['index']==pid]['x1'])
        x2 = np.array(data[data['index']==pid]['x2'])
        x.append(  0.5*(x1+x2)  )
        y1 = np.array(data[data['index']==pid]['y1'])
        y2 = np.array(data[data['index']==pid]['y2'])
        y.append( -0.5*(y1+y2)  )
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

def get_transformation_to_reference( reference_points, target_points )
    #returns parameters to an transformation
    M = np.zeros( (6,6) )
    M[0,0:2] = target_points[0,:]
    M[1,0:2] = target_points[1,:]
    M[2,0:2] = target_points[2,:]
    M[3,2:4] = target_points[0,:]
    M[4,2:4] = target_points[1,:]
    M[5,2:4] = target_points[2,:]
    M[0,4] = 1.
    M[1,4] = 1.
    M[2,4] = 1.
    M[3,5] = 1.
    M[4,5] = 1.
    M[5,5] = 1.
    from scipy.linalg import solve
    return list( solve( M , reference_points.flatten() ) )

if __name__ == '__main__':
    print "Let us test this baby"
    reference_points = pd.read_csv('../annotations/deathCircle/video0/reference_points.csv').as_matrix()
    origin = reference_points[0,:]
    reference_points -= origin
    #For the deathCircle videos the resolution is 1630 x 1948
    #we normalist to the square [-1 1 -1 1]
    reference_points[:,0] = 2*reference_points[:,0]/float(1630) - 1
    reference_points[:,1] = 2*reference_points[:,0]/float(1948) - 1
    base_directory = '../annotations/deathCircle/'
    for folder in ['video0/','video1/','video2/','video3/','video4/']:
        fname = base_director + folder + 'annotations.txt'
        x_raw,y_raw = get_trajectories( fname )
        #NOW YOU TRANFORM THEM TO REF COORDINATES HERE
        target_points = pd.read_csv( base_directory+folder+'reference_points.csv').as_matrix()
        x_list,vx_list,ax_list,y_list,vy_list,ay_list = smooth_trajectories(x_raw, y_raw)
        a,b,c,d,e,f = get_transformation_to_reference( target_points )
        x_arr = map( lambda x,y: a*x+b*y+e , x_list, y_list)
        y_arr = map( lambda x,y: c*x+d*y+f , x_list, y_list)
        vx_arr = map( lambda x,y: a*x+b*y , vx_list, vy_list )
        vy_arr = map( lambda x,y: c*x+d*y , vx_list, vy_list )
        ax_arr = map( lambda x,y: a*x+b*y , ax_list, ay_list )
        ay_arr = map( lambda x,y: c*x+d*y , ax_list, ay_list )
        functionals = []
        for traj_tuple in zip(x_arr,vx_arr,ax_arr,y_arr,vy_arr,ay_arr):
            for arg in zip( *traj_tuple):
                functionals += EL_functional_2d(*arg)
        functionals_npy = np.stack( functionals )
        from time import time
        t0 = time()
        Q = np.einsum('ki,kj->ij', functionals_npy , functionals_npy )
        total_time = time() - t0
        print "Computing Q from ells tooks %f seconds" % total_time
    np.save( "Q_matrix")

