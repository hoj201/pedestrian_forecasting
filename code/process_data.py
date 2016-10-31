import numpy as np

def get_BB_ts_list( folder , label='Pedestrian' ):
    """ Returns trajectories and height and width of a domain

    args:
        folder: string of folder name
    
    kwargs:
        label: string of label name.  Options are "Pedestrian" and "Biker"

    returns:
        BB_ts_list, width, height
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

    # RESCALE SO WIDTH = 1.0
    offset = np.array( [ 0.5, height/(2*float(width)), 0.5, height/(2*float(width)) ] ).reshape(4,1)
    transformation = lambda BB_ts : BB_ts / float(width) - offset
    BB_ts_list = map( transformation, BB_ts_list )
    width, height = 1.0, height/ float(width)

    #Filter out trajectories that do not enter the scene at the boundary
    def enters_and_leaves( BB_ts ):
        outside = lambda left, bottom, right, top: left < -0.9*width/2 or right > 0.9*width/2 or bottom < -0.9*height/2 or top > 0.9*height/2
        enters = outside( * list( BB_ts[:,0] ) )
        leaves = outside( * list( BB_ts[:,-1] ) )
        return enters and leaves
    BB_ts_list = filter( enters_and_leaves, BB_ts_list )

    #Filters for trajectories who's endpoints are close
    def distant_endpoints( BB_ts ):
        dx = BB_ts[0,0]- BB_ts[0,-1]
        dy = BB_ts[1,0]- BB_ts[1,-1]
        return dx**2 + dy**2 > width*height*0.01
    BB_ts_list = filter( distant_endpoints, BB_ts_list )

    #Remove outliers with respect to length based on IQR criterion
    def length( BB_ts ):
        x,y = BB_ts_to_curve( BB_ts )
        n = len(x)
        u = x[1:] - x[:n-1]
        v = y[1:] - y[:n-1]
        return np.sqrt( u**2 + v**2 ).sum()

    length_list = map( length , BB_ts_list )
    length_list.sort()
    from matplotlib import pyplot as plt
    plt.hist( length_list )
    plt.show()
    n = len( length_list )
    print "n = {n}".format(n=n)
    Q1, Q3 = length_list[ n / 4], length_list[ 3*n / 4]
    IQR = Q3 - Q1
    BB_ts_list = filter( lambda BB_ts: length( BB_ts ) < Q3+1.5*IQR and length( BB_ts ) > Q1 - 1.5*IQR , BB_ts_list)
    return BB_ts_list, width, height

def get_bbox_width( BB_ts_ls ):
    """ computes the maximum observed bounding box width in a bounding box time_series
    """
    width_f = lambda BB_ts: ( BB_ts[2] - BB_ts[0] ).max()
    height_f = lambda BB_ts: ( BB_ts[3] - BB_ts[1] ).max()
    max_width = max( map( width_f , BB_ts_ls ) )
    max_height = max( map( height_f, BB_ts_ls ) )
    return max( max_width, max_height )

def get_bbox_velocity_width( BB_ts_ls ):
    """ computes the maximum observed velocity bbox width in a bounding box time_series
    """
    v_function = lambda x: (x[2:] - x[:len(x)-2])/2.0
    width_function = lambda BB_ts: np.abs( v_function( BB_ts[2] ) - v_function( BB_ts[0] ) ).max()
    height_function = lambda BB_ts: np.abs( v_function( BB_ts[3] ) - v_function( BB_ts[1] ) ).max()
    max_width = max( map( width_function, BB_ts_ls ) )
    max_height = max( map( height_function, BB_ts_ls ) )
    return max( max_width, max_height )

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

def get_std_velocity( BB_ts_ls ):
    """  Computes the standard deviation of the velocity measurements
    """
    def BB_ts_to_std( BB_ts ):
        x = 0.5*( BB_ts[0] + BB_ts[2] )
        y = 0.5*( BB_ts[1] + BB_ts[3] )
        u = (x[2:] - x[:len(x)-2]) / 2.0
        v = (y[2:] - y[:len(y)-2]) / 2.0
        return np.sqrt( u.std()**2 + v.std()**2 )
    return max( map( BB_ts_to_std , BB_ts_ls ) )

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    BB_ts_list, width, height = get_BB_ts_list("../annotations/coupa/video2/", label = "Biker")
    for BB_ts in BB_ts_list:
        curve = BB_ts_to_curve( BB_ts )
        plt.plot( curve[0], curve[1], 'b-' )
    plt.axis('equal')
    plt.show()

    w = get_bbox_width( BB_ts_list)
    print "max width = {}".format(w)
