import numpy as np
from matplotlib import pyplot as plt
from scene import scene

#--------------------------------------------------------------------------------
# CHOOSE A VIDEO, LEARN CLASSES
#--------------------------------------------------------------------------------
import process_data
folder = '../annotations/coupa/video2/'
x_data, y_data, V_scale = process_data.get_trajectories(folder, label="Biker")
print "Loaded %d trajectories.\n" % len(x_data)

for x,y in zip(x_data, y_data):
    plt.plot( x, y, 'b-' )
plt.axis('equal')
plt.show()


#Aggregate the data into a training and a test set
curve_ls = [ np.vstack([x,y]) for (x,y) in zip( x_data, y_data ) ]

from os.path import isfile
import pickle
if isfile('learned_scene.pkl'):
    file_in1 = open('learned_scene.pkl','rb')
    file_in2 = open('test_set.pkl','rb')
    learned_scene = pickle.load( file_in1 )
    test_set = pickle.load( file_in2 )
    file_in1.close()
    file_in2.close()
    print "Loaded learned scene"
else:
    file_out1 = open('learned_scene.pkl','wb')
    file_out2 = open('test_set.pkl','wb')
    from sklearn.cross_validation import train_test_split
    train_set, test_set = train_test_split( curve_ls, random_state = 0 )

    #Train a scene
    print "Training"
    learned_scene = scene( train_set, V_scale )
    print "Training complete."
    pickle.dump( learned_scene, file_out1,-1) 
    pickle.dump( test_set, file_out2,-1) 
    file_out1.close()
    file_out2.close()

#--------------------------------------------------------------------------------
# DISPLAY CLUSTERS
#--------------------------------------------------------------------------------
for k in range( learned_scene.num_nl_classes ):
    from visualization_routines import visualize_cluster
    visualize_cluster( learned_scene, k )

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
x_min = V_scale[0]/float(6) - V_scale[0]
x_max = -x_min
y_min = V_scale[1]/float(6) - V_scale[1]
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
print "s_max = %f" % learned_scene.s_max

helper = lambda x,y : learned_scene.P_of_x_given_mu( np.array([x,y]) )
X_grid, Y_grid = np.meshgrid(
        np.linspace( -learned_scene.V_scale[0], learned_scene.V_scale[0], 40),
        np.linspace( -learned_scene.V_scale[1], learned_scene.V_scale[1], 40)
        )

rho_grid = helper(X_grid, Y_grid )
cs = plt.contourf( X_grid, Y_grid, rho_grid, 50, cmap = 'viridis' )
plt.colorbar(cs)
plt.axis('equal')
plt.show()



# TEST ADVECTION ROUTINE ON A SINGLE VF
dynamics = lambda x,jac=False: learned_scene.director_field_vectorized( 1, x, jac=jac)
from particle_advect import advect_vectorized as advect
x0,y0 = X_grid.flatten(), Y_grid.flatten()
t_arr = np.array( [0.0, -0.5] )
x_t,y_t,w_t = advect( dynamics, x0, y0 , -0.5, 2 )

rho_0 = helper( x_t[0], y_t[0] ) * w_t[0]
rho_0 = rho_0.reshape( X_grid.shape )
rho_1 = helper( x_t[1], y_t[1] ) * w_t[1]
rho_1 = rho_1.reshape( X_grid.shape )

fig, ax_arr = plt.subplots( 3,1,figsize=(5,10) )
ax_arr[0].contourf( X_grid, Y_grid, rho_0, 50, cmap='viridis')
ax_arr[0].axis('equal')
ax_arr[0].set_title('t = %f' % t_arr[0] )
ax_arr[2].contourf( X_grid, Y_grid, rho_1, 50, cmap='viridis')
ax_arr[2].axis('equal')
ax_arr[2].set_title('t = %f' % t_arr[1] )

UV = dynamics( np.vstack( [X_grid.flatten(), Y_grid.flatten() ] ) )
U_grid = UV[0].reshape( X_grid.shape)
V_grid = UV[1].reshape( X_grid.shape)
ax_arr[1].quiver( X_grid, Y_grid, U_grid, V_grid , pivot='mid', scale=50.)
ax_arr[1].axis('equal')

fig.suptitle('Advection for a single vector-field')
plt.show()

print "Testing class prediction routine"
T = 10.0
rho_T = learned_scene.predict_pdf( X_grid, Y_grid, T )
fig, ax_arr = plt.subplots( 2,1,figsize=(5,10) )
ax_arr[0].contourf( X_grid, Y_grid, rho_0, 50, cmap='viridis')
ax_arr[0].axis('equal')
ax_arr[1].contourf( X_grid, Y_grid, rho_T , 50, cmap = 'viridis')
ax_arr[1].axis('equal')
plt.show()
