import numpy as np

def acceleration_to_cocoefs( x,y,vx, vy, ax, ay ):
    #generates two vectors of co-coefficients for the theta parameters
    #TODO: make this work and vectorize it.
    return [q1,q2]

def get_quadratic_form( two_jet_data ):
    #two_jet_data is a list of tupes (x,y,vx,vy,ax,ay)
    for i in range( len( two_jet_data ) ):
        q1,q2 = acceleration_to_cocoefs(x[i],y[i],
