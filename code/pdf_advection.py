import numpy as np
from matplotlib import pyplot as plt
from hermite_function import hermite_function_series as h_series
from hermite_function import FP_operator
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
# LEARN FOKKER PLANCK OPERATORS
#--------------------------------------------------------------------------------
print "Building Fokker Planck Operators"
deg = (50, 50)
FP_ls = []
from director_field import director_field_to_FP_operator as df2FP
k_max = learned_scene.k_max_theta
for k in range( learned_scene.num_nl_classes ):
    FP_ls.append( df2FP( learned_scene.theta_coeffs[k], k_max, V_scale,  deg=deg, poly_deg=(5,5) ) )

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
x_min = V_scale[0]/5 - V_scale[0]
x_max = -x_min
y_min = V_scale[1]/5 - V_scale[1]
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

rho_0_linear = h_series( M=V_scale, deg=deg )
helper = lambda x,y : learned_scene.P_of_x_given_mu( np.array([x,y]) )
rho_0_linear.interpolate( helper )
X_grid, Y_grid = np.meshgrid(
        np.linspace( -V_scale[0], V_scale[0], 40 ),
        np.linspace( -V_scale[1], V_scale[1], 40 )
        )

rho_grid = rho_0_linear.evaluate_on_grid( [X_grid, Y_grid] )
cs = plt.contourf( X_grid, Y_grid, rho_grid, 50, cmap = 'viridis' )
plt.colorbar(cs)
plt.show()

# INITIALIZE out = 0  (type=h_series) FOR OUTPUT
rho_T = h_series( M=V_scale, deg=deg )
T = 0.005

from scene import sigma_x, sigma_v, sigma_s
eta_mag = np.sqrt( eta_test[0]**2 + eta_test[1]**2 )
s_ls = np.linspace( -5*sigma_s, 5*sigma_s , 30 )
# FOR EACH CLASS, SPEED FIND INITIAL CONDITION TO ADVECT
from itertools import product
for k,s in product( range( learned_scene.num_nl_classes ), s_ls):
    P_cs = learned_scene.P_of_nl_class_and_speed_given_measurements(k,s)
    helper = lambda x,y: P_cs*learned_scene.P_of_x_given_measurements_nl_class_speed( np.array([x,y]) , k, s)
    rho_0 = h_series( M=V_scale, deg=deg )
    rho_0.interpolate( helper )

    #SUBTRACT FROM MASS DEDICATED TO LINEAR CLASS
    rho_0_linear -= rho_0

    #ADVECT AND ADD TO OUT, WEIGHTED BY P(c,s|mu,eta)
    rho_T += FP_ls[k].advect( rho_0,  T / s)
    print "k=%d, s=%f, P(c,s|mu,eta) = %f" % (k,s,P_cs)


# ADD LINEAR TERM
#... something with rho_0_linear

# DISPLAY RESULTS
rho_grid = rho_0_linear.evaluate_on_grid( [X_grid, Y_grid] )
cs = plt.contourf( X_grid, Y_grid, rho_grid, 50, cmap='viridis')
plt.colorbar(cs)
plt.show()
