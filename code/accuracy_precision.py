import numpy as np
import pickle
from scene import Scene
from generate_distributions import make_generator

def integrate_bbox(BBox, xy, weights):
    """ Computes the integral of rho over a bounding box. """
    x_min = BBox[0]
    y_min = BBox[1]
    x_max = BBox[2]
    y_max = BBox[3]
    x,y = xy
    # Only sum weights of Dirac-deltas contained within the bbox
    out = np.dot( weights, (x > x_min)*(x < x_max)*(y > y_min)*(y < y_max) )
    return out

################################################################################
#   Load scene and test set
################################################################################
with open('test_scene.pkl', 'rb') as f:
    scene = pickle.load(f)
with open('test_set.pkl', 'rb') as f:
    test_set = pickle.load(f)
dt = 0.1
Nt = 10
test_BB_ts = test_set[3]
from matplotlib import pyplot as plt
plt.plot(test_BB_ts[0], test_BB_ts[1])
plt.axis([-scene.width, scene.width, -scene.height, scene.height])
plt.axis('equal')
plt.show()
def get_initial_condition(BB_ts):
    fd_width = 4
    BB0 = BB_ts[:,0]
    BB2 = BB_ts[::,,fd_width]
    x = 0.5*(BB0[0]+BB0[2]+BB2[0]+BB2[2]) / fd_width
    y = 0.5*(BB0[1]+BB0[3]+BB2[1]+BB2[3]) / fd_width
    u = 0.5*(BB2[0]-BB0[0]+BB2[2]-BB0[2]) / fd_width
    v = 0.5*(BB2[1]-BB0[1]+BB2[3]-BB0[3]) / fd_width
    return np.array([x, y]), np.array([u, v])

x,v = get_initial_condition(test_BB_ts[:, 10:])
speed = np.sqrt(np.sum(v**2))
print "Measured speed / sigma_v = {:f}".format( speed / scene.sigma_v )
print "sigma_v = {:f}".format( scene.sigma_v)


################################################################################
#   Compute Accuracy and Precision
################################################################################
gen = make_generator(scene, x, v, dt, Nt)
for data in gen:
    for xy,weights in data:
        integrate_bbox(BBox, xy, weights)
