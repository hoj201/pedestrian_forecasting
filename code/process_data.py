import numpy as np

def get_BB_ts_list( folder , label='Pedestrian' ):
    """ Returns trajectories and height and width of a domain

    args:
        folder: string of folder name
    
    kwargs:
        label: string of label name.  Options are "Pedestrian" and "Biker"

    returns:
        BB_ts_list, V_scale
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
    from PIL import Image
    fname = folder + 'reference.jpg'
    im = Image.open(fname)
    width,height = im.size
    ids = set( data[ data['label']==label ]['id'])
    BB_ts_list = [ data[ data['id'] == id ].loc[:,'x1':'y2'].values.transpose() for id in ids ]
    offset = np.array( [ 1.0, height/float(width), 1.0, height/float(width) ] ).reshape(4,1)
    transformation = lambda BB_ts : 2*BB_ts / float(width) - offset
    BB_ts_list = map( transformation, BB_ts_list )
    V_scale = (1.0 , height/float(width) )

    #Filter out trajectories that do not enter the scene at the boundary
    def enters_and_leaves( BB_ts ):
        outside = lambda left, bottom, right, top: left < -0.9*V_scale[0] or right > 0.9*V_scale[0] or bottom < -0.9*V_scale[1] or top > 0.9*V_scale[1]
        enters = outside( * list( BB_ts[:,0] ) )
        leaves = outside( * list( BB_ts[:,-1] ) )
        return enters and leaves

    #Filters for trajectories who's endpoints are close
    def distant_endpoints( BB_ts ):
        dx = BB_ts[0,0]- BB_ts[0,-1]
        dy = BB_ts[1,0]- BB_ts[1,-1]
        return dx**2 + dy**2 > V_scale[0]*V_scale[1]*0.2

    BB_ts_list = filter( distant_endpoints, filter( enters_and_leaves, BB_ts_list ) )

    #Remove outliers with respect to log of length based on IQR criterion
    def log_length( BB_ts ):
        x,y = BB_ts_to_curve( BB_ts )
        n = len(x)
        u = x[1:] - x[:n-1]
        v = y[1:] - y[:n-1]
        return np.log( np.sqrt( u**2 + v**2 ).sum() )

    length_list = map( log_length , BB_ts_list )
    length_list.sort()
    n = len( length_list )
    Q1, Q3 = length_list[ n / 4], length_list[ 3*n / 4]
    IQR = Q3 - Q1
    BB_ts_list = filter( lambda BB_ts: log_length( BB_ts ) < Q3+1.5*IQR and log_length( BB_ts ) > Q1 - 1.5*IQR , BB_ts_list)
    return BB_ts_list, V_scale

def BB_ts_to_curve( BB_ts ):
    """ Converts a BB_ts to a curve

    """
    transformation = np.array([[0.5, 0.0, 0.5, 0.0] , [0.0, 0.5, 0.0, 0.5] ] )
    return np.einsum( 'ik,kj', transformation, BB_ts )

def BB_to_position_and_velocity( BB_0, BB_1, delta_t = 1 ):
    x = 0.25* ( BB_0[0] + BB_0[2] + BB_1[0] + BB_1[2] )
    y = 0.25* ( BB_0[1] + BB_0[3] + BB_1[1] + BB_1[3] )
    position = np.array( [x, y] )
    dx = 0.5* ( (BB_1[0] - BB_0[0] ) + (BB_1[2] - BB_0[2] ) ) / float( delta_t )
    dy = 0.5* ( (BB_1[1] - BB_0[1] ) + (BB_1[3] - BB_0[3] ) ) / float( delta_t )
    velocity = np.array( [dx, dy] )
    return position, velocity

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    BB_ts_list, V_scale = get_BB_ts_list("../annotations/coupa/video2/", label = "Biker")
    for BB_ts in BB_ts_list:
        curve = BB_ts_to_curve( BB_ts )
        plt.plot( curve[0], curve[1], 'b-' )
    plt.axis('equal')
    plt.show()


