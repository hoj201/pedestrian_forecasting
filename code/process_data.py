# LOAD DATA
import pandas as pd
directory_name = "../annotations/deathCircle/video0/"
file_name = "annotations.txt"
train = pd.read_csv(directory_name+file_name, sep=" ",\
        usecols=[0,1,2,3,4,9])

train.columns = ['index','x1','y1','x2','y2','label']

print "Data loaded"
print train.head()

# GET POSITIONS
x=[]
y=[]
pedestrian_indices = set(train[ train['label']=='Pedestrian' ]['index'])
import numpy as np
for pid in pedestrian_indices:
    x1 = np.array(train[train['index']==pid]['x1'])
    x2 = np.array(train[train['index']==pid]['x2'])
    x.append( 0.5*(x1+x2) )
    y1 = np.array(train[train['index']==pid]['y1'])
    y2 = np.array(train[train['index']==pid]['y2'])
    y.append( 0.5*(y1+y2) )

from matplotlib import pyplot as plt
im = plt.imread(directory_name+"reference.png")
implot = plt.imshow(im)
ax=[]
ay=[]
for x_traj,y_traj in zip(x,y):
    from scipy import signal
    sigma =10.0
    G = signal.gaussian(100,sigma)
    G = G / G.sum()
    x_smooth = signal.fftconvolve( x_traj,G, mode='valid')
    y_smooth = signal.fftconvolve( y_traj,G, mode='valid')
    plt.plot(x_smooth, y_smooth,'b-')
    N = x_smooth.size
    ax.append( -x_smooth[0:(N-2)] + 2*x_smooth[1:(N-1)]-x_smooth[2:N] )
    ay.append( -y_smooth[0:(N-2)] + 2*y_smooth[1:(N-1)]-y_smooth[2:N] )

plt.show()
