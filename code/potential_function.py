import numpy as np

#**************************************
#********** RETRIEVE AND PROCESS IMAGES
#**************************************

directory = "../annotations/deathCircle/video0/"
terrain_types = ["buildings", "grass", "street", "sidewalk"]
from scipy import misc
from scipy.signal import decimate, fftconvolve, gaussian
downsample_rate = 4
#reader translates an file-name to an image into a boolean array (i.e. a BW image)
reader = lambda fname : (misc.imread(directory+fname)).min(axis=2) == 0
decimater = lambda x: decimate( decimate( x , downsample_rate , axis=0), downsample_rate, axis=1)

normal_dist = lambda sigma,N_g: gaussian(N_g,sigma) / (N_g*np.sqrt(2*np.pi*sigma**2))
kernel = lambda sigma,N_g: np.outer( normal_dist(sigma,N_g) , normal_dist(sigma,N_g) )
smoother = lambda im,sigma : fftconvolve( im , kernel(sigma,int(3*sigma)), mode='valid')
get_smooth_image = lambda fname,sigma: smoother( decimater( reader( fname ) ), sigma )

sigmas = [5,10,20]

#terrain_image[t][s] is an image of terrain t convolved with a gaussian of scale s in sigmas.
terrain_image = { t: {s: get_smooth_image( t + ".bmp",s ) \
        for s in sigmas} \
        for t in terrain_types}


#X_res[t][s] is the X-resolution of terrain_image[t][s]
X_res = {t: {s: terrain_image[t][s].shape[0] for s in sigmas} for t in terrain_types}
Y_res = {t: {s: terrain_image[t][s].shape[1] for s in sigmas} for t in terrain_types}

#Now we have a dictionary of processed images.  Lets FFT them.
#terrain_image_fft[t][s] is an the fourier transform of terrain[t][s]
terrain_image_fft = { t: {s: np.fft.fftn(terrain_image[t][s]) \
        for s in sigmas} \
        for t in terrain_types}


#****************************************
#************* SAVE THE LOW FOURIER MODES
#****************************************

i_max = 20
j_max = 20
indices = [(i,j)    for i in range(-i_max,i_max+1) \
                    for j in range(-j_max,j_max+1)]

#coeffs[t][s][ij] is the ij^th Fourier coefficient of terrain_image[t][s].
coeffs = {t: {s: {ij: terrain_image_fft[t][s][ij[0],ij[1]] for ij in indices} for s in sigmas} for t in terrain_types}


#*****************************************
#************ CONSTRUCT POTENTIAL FUNCTION
#*****************************************

e2pij = lambda i,j,x,y : np.exp(2*np.pi*1j*(i*x+j*y))

def V(x,y,theta):
    #theta should be a dict of dicts such that theta[t][s] is a float
    global terrain_types, coeffs, indices, sigmas
    if type(x) in { type(1) , type(1.0) }:
        out = 0.0
    else:
        out = np.zeros( x.shape )
    from itertools import product
    for (t,s) in product( terrain_types , sigmas):
        N = float( X_res[t][s] * Y_res[t][s] )
        for ij in indices:
            (i,j) = ij
            out += theta[t][s]*(coeffs[t][s][ij]*e2pij(i,j,x,y) ).real / N
    return out
