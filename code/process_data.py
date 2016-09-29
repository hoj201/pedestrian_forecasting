import numpy as np

def get_trajectories( folder , label='Pedestrian' ):
    """ Returns trajectories and height and width of a domain

    args:
        folder: string of folder name
    
    kwargs:
        label: string of label name.  Options are "Pedestrian" and "Biker"

    returns:
        x_data, y_data, width, height
    """
    import pandas as pd

    data = pd.read_csv( folder + 'annotations.txt', sep=" ")
    data.columns = ['id',
            'x1','y1','x2','y2',
            'frame',
            'lost','occluded','generated',
            'label']
    # GET POSITIONS
    data = data[ data.lost != 1]

    x_raw=[]
    y_raw=[]
    relevant_indices = set( data[ data['label']==label ]['id'])
    import numpy as np
    for pid in relevant_indices:
        x1 = np.array(data[data['id']==pid]['x1'])
        x2 = np.array(data[data['id']==pid]['x2'])
        x_raw.append(  0.5*(x1+x2)  )
        y1 = np.array(data[data['id']==pid]['y1'])
        y2 = np.array(data[data['id']==pid]['y2'])
        y_raw.append( 0.5*(y1+y2)  )
    from PIL import Image
    fname = folder + 'reference.jpg'
    im = Image.open(fname)
    width,height = im.size
    x_data = map( lambda x: 2*x/width-1.0 , x_raw )
    y_data = map( lambda y: 2*y/width-height/float(width) , y_raw )
    V_scale = (1.0 , height/float(width) )
    x_data, y_data = filter_trajectories( x_data, y_data, V_scale )
    return x_data, y_data, V_scale

def filter_trajectories( x_raw, y_raw, V_scale ):
    """ Removes any trajectories that do not leave the scene or are small.

    args:
        x_raw:
        y_raw:
        V_scale: iterable, length 2.

    returns:
        x_out:
        y_out:
    """
    inside = lambda x,y : np.abs(x) < 0.9*V_scale[0] and np.abs(y) < 0.9*V_scale[1]
    small = lambda x,y : np.sqrt( (x[0]-x[-1])**2 + (y[0] - y[-1])**2 ) < np.sqrt(V_scale[1]*V_scale[0])*0.20
    x_out, y_out = [], []
    for x,y in zip( x_raw, y_raw):
        if inside( x[0], y[0] ) or inside( x[-1], y[-1] ) or small(x,y):
            continue
        x_out.append(x)
        y_out.append(y)
    return x_out, y_out

def prune( x_data, y_data ):
    """ Removes the abnormally long/short trajectories

    args:
        x_data:
        y_data:

    returns:
        n_discarded:
        x_data_pruned:
        y_data_pruned:
    """
    def length_of_traj( x, y ):
        n = len(x)
        u = x[1:] - x[:n-1]
        v = y[1:] - y[:n-1]
        return np.sqrt(u**2+v**2).sum()
    n = len(x_data)
    lengths = map( length_of_traj , x_data, y_data )
    lengths.sort()
    #Compute IQR
    Q1 = lengths[n/4]
    Q3 = lengths[3*n/4]
    IQR = Q3-Q1

    #Compute which to discard and count hoe many agents you discard
    keep_it = lambda x,y: length_of_traj(x,y) < Q3+1.5*IQR and length_of_traj(x,y) > Q1-1.5*IQR
    bool2int = lambda b: 1 if b else 0
    n_keep = reduce( lambda x,y:x+y, map( bool2int, map( keep_it, x_data, y_data ) ) )
    n_discarded = len(x_data) - n_keep
    return n_discarded, zip( *filter( keep_it , zip(x_data, y_data) ))

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


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    x_data, y_data, V_scale = get_trajectories("../annotations/coupa/video2/", label="Biker")
    from matplotlib import pyplot as plt
    for x,y in zip(x_data, y_data):
        plt.plot( x, y, 'b-' )
    plt.axis('equal')
    plt.show()
