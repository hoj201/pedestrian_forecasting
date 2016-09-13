import numpy as np
from matplotlib import pyplot as plt
from scene import scene

#--------------------------------------------------------------------------------
# CHOOSE A VIDEO, LEARN CLASSES
#--------------------------------------------------------------------------------
import process_data
process_data = reload(process_data)
folder = '../annotations/coupa/video2/'
fname = folder + 'annotations.txt'
x_raw,y_raw = process_data.get_trajectories(fname,label="Biker")
from PIL import Image
fname = folder + 'reference.jpg'
im = Image.open(fname)
width,height = im.size
x_data = map( lambda x: x-width/2 , x_raw )
y_data = map( lambda x: x-height/2 , y_raw )

#resize the curves and set V_scale
x_data = map( lambda x: 2*x/float(width) , x_data )
y_data = map( lambda x: 2*x/float(width) , y_data ) #Not a typo.
V_scale = (1.0, height / float(width) )

#Aggregate the data into a training and a test set
curve_ls = [ np.vstack([x,y]) for (x,y) in zip( x_data, y_data ) ]
from sklearn.cross_validation import train_test_split
train_set, test_set = train_test_split( curve_ls, random_state = 0 )

#Train a scene
print "Training"
learned_scene = scene( train_set, V_scale )
print "Training complete."


#--------------------------------------------------------------------------------
# CHOOSE A TRAJECTORY
#--------------------------------------------------------------------------------
test_curve = test_set[0]
plt.plot( test_curve[0], test_curve[1], 'b-')
plt.plot( test_curve[0,0], test_curve[1,0], 'ro')
plt.axis( [-V_scale[0], V_scale[0], -V_scale[1], V_scale[1] ] )
plt.show()


# INITIALIZE MEASUREMENTS (mu,eta)

#CHOOSE INITIAL CONDITIONS SO THAT THEY ARE WITHIN THE RESOLVED AREA
x_min = V_scale[0]/float(5) - V_scale[0]
x_max = -x_min
y_min = V_scale[1]/float(5) - V_scale[1]
y_max = -y_min
in_region = lambda x,y: x > x_min and x< x_max and y>y_min and y<y_max

i = 0
fd_width = 2
x,y = list( test_curve[:,i+fd_width] )
while not in_region(x,y):
    i += 1
    x,y = list( test_curve[:,i+fd_width] )

mu_test = test_curve[:,i+fd_width]
eta_test = (test_curve[:,i+2*fd_width] - test_curve[:,i]) / (2*fd_width)
learned_scene.set_mu( mu_test )
learned_scene.set_eta( eta_test )

print "mu = (%f, %f)" % tuple(mu_test)
print "eta = (%f, %f)" % tuple(eta_test)

helper = lambda x,y : learned_scene.P_of_x_given_mu( np.array([x,y]) )
X_grid, Y_grid = np.meshgrid(
        np.linspace( -V_scale[0], V_scale[0], 40 ),
        np.linspace( -V_scale[1], V_scale[1], 40 )
        )

rho_grid = helper(X_grid, Y_grid )
cs = plt.contourf( X_grid, Y_grid, rho_grid, 50, cmap = 'viridis' )
plt.colorbar(cs)
plt.show()

T = 0.005
quit()
from scene import sigma_x, sigma_v
s_ls = np.linspace( -learned_scene.s_max, learned_scene.s_max, 30 )
ds = s_ls[1] - s_ls[0]
# FOR EACH CLASS ADVECT FOR TIME T/S AND ADD TO OUTPUT
for k in range( learned_scene.num_nl_classes ):
    P_cs = [ learned_scene.P_of_nl_class_and_speed_given_measurements(k,s) for s in s_ls ]
    P_cs = np.array( P_cs )
    P_c = P_cs.sum() * ds
    tol = 1e-3
    if P_c < tol:
        print "Skipping computation for class c_%d.  P(c_%d | mu )=%f < %f \n" % (k,k,P_c,tol)
        continue
    from particle_advect import advect
    nodes = np.vstack( [X_grid.flatten() , Y_grid.flatten()] )
    weights_0 = learned_scene.P_of_x_given_mu( nodes )
    dynamics = lambda x,jac=False: learned_scene.director_field( k, x, jac=jac)
    weights_t = advect( dynamics, nodes, weights_0, t_span) #weights_t is of shape (N_t, nodes.size )

    #SUBTRACT FROM MASS DEDICATED TO LINEAR CLASS
    #rho_0_linear -= rho_0

    #ADVECT AND ADD TO OUT, WEIGHTED BY P(c,s|mu,eta)
    rho_T += np.dot( P_cs, weights_t ) * ds
    print "c_%d case computed.\n" % k


# ADD LINEAR TERM
#... something with rho_0_linear

# DISPLAY RESULTS
cs = plt.contourf( X_grid, Y_grid, rho_T, 50, cmap='viridis')
plt.colorbar(cs)
plt.show()
