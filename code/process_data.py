def get_trajectories( fname , label='Pedestrian' ):
    #a function to load trajectories from annotation data.
    # LOAD DATA
    import pandas as pd

    data = pd.read_csv( fname, sep=" ")
    data.columns = ['id',
            'x1','y1','x2','y2',
            'frame',
            'lost','occluded','generated',
            'label']
    # GET POSITIONS
    data = data[ data.lost != 1]

    x=[]
    y=[]
    relevant_indices = set( data[ data['label']==label ]['id'])
    import numpy as np
    for pid in relevant_indices:
        x1 = np.array(data[data['id']==pid]['x1'])
        x2 = np.array(data[data['id']==pid]['x2'])
        x.append(  0.5*(x1+x2)  )
        y1 = np.array(data[data['id']==pid]['y1'])
        y2 = np.array(data[data['id']==pid]['y2'])
        y.append( 0.5*(y1+y2)  )
    return [x,y]

def smooth_trajectories( x_raw , y_raw ):
    #Smooths and extracts positions, velocities, and accelerations
    x_list = []
    y_list = []
    vx_list = []
    vy_list = []
    N_points = 0
    sigma =5.0
    from scipy import signal
    G = signal.gaussian(5*sigma,sigma)
    G = G / G.sum()
    for x_traj,y_traj in zip(x_raw,y_raw):
        x_smooth = signal.fftconvolve( x_traj, G , mode='valid' )
        y_smooth = signal.fftconvolve( y_traj, G , mode='valid' )
        N = x_smooth.size
        x_list.append( x_smooth[1:(N-1)] )
        y_list.append( y_smooth[1:(N-1)] )
        vx_list.append( 0.5*(-x_smooth[0:(N-2)] + x_smooth[2:N]) )
        vy_list.append( 0.5*(-y_smooth[0:(N-2)] + y_smooth[2:N]) )
    return x_list, vx_list, y_list, vy_list

import numpy as np
def display_trajectories(x_list,y_list,scene):
    from matplotlib import pyplot as plt
    directory_name = "../annotations/" + scene + "/video0/"
    im = plt.imread(directory_name+"reference.png")
    implot = plt.imshow(im, extent=[-1,1,-1,1])
    for x,y in zip(x_list,y_list):
        plt.plot(x,y,'b-')
    plt.show()
    return 0

def get_transformation_to_reference( reference_points, target_points ):
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
    from matplotlib import pyplot as plt
    x,y = get_trajectories("../annotations/coupa/video0/annotations.txt", label="Biker")
    plt.plot(x[3])
    plt.show()
