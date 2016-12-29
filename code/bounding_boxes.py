import numpy as np 
def eval_bounding_boxes(rho, bboxes):
    """
    Takes:
    #bboxes: np.array(n_boxes, 2, 2)
    ##looks like [[[box_width, box_height], [box_x, box_y]]]
    #rho: (np.array(n_points, 2), np.array(n_points))
    #returns: float array , 
    """
    xy, p = rho
    asdf = {"counter": 0}
    def filter(e): #NOTE hoj:  Probably should rename this, as filter is a built-in func
        fn = lambda x: (np.absolute((x[0]-e[1][0])) <= e[0][0]/2.0 + 10E-6) and \
                       (np.absolute(x[1] - e[1][1]) <= e[0][1]/2.0 + 10E-6)
        asdf["counter"] += 1
        print "{}%".format(100*asdf["counter"] / float(len(E)))
        return sum(p[np.where(map(fn, xy))[0]])
    res = np.array(map(filter, E))
    return res

def bboxes_from_bounds(bounds, resolution):
    """
    Takes:
    #bounds: (x_min, x_max, y_min, y_max)
    #resolution (x, y)
    Returns:
    #bboxes:
    ##looks like: [[[box_width, box_height], [box_x, box_y]]]
    """
    x_low = np.linspace(x_min, x_max, resolution[0]+1)
    x_high = x_low + (x_max-x_min)/resolution[0]
    xs = ((x_low + x_high)/2)[:-1]
    y_low = np.linspace(y_min, y_max, resolution[1]+1)
    y_high = y_low + (y_max-y_min)/resolution[1]
    ys = ((y_low + y_high)/2)[:-1]

    xs, ys = np.meshgrid(xs, ys)
    width_x = (x_max - x_min)/resolution[0]
    width_y = (y_max - y_min)/resolution[1]

    pts = [list(x) for x in zip(xs.flatten(), ys.flatten())]
    sizes = [list(x) for x in zip(np.ones(len(pts)) * width_x,
                                  np.ones(len(pts)) * width_y)]
    return [list(x) for x in zip(sizes, pts)]
