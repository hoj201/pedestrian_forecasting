from sys import argv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from data import sets, scenes
scene = int(argv[1])
folder = argv[2]
result = argv[3]
reference = argv[4]
width = int(argv[5])
height = int(argv[6])

for file in os.listdir(folder):
    if "DS" in file: continue
    print file
    data = np.load(folder + file)
    print data.shape
    ax = plt.gca()
    plt.axis('off')
    plt.imshow(data.reshape(width, height).transpose(), origin="lower", extent=[-width/2.0,width/2.0,-height/2.0,height/2.0], cmap="viridis")
    plt.imshow(mpimg.imread(reference), extent=[-width/2.0, width/2.0,-height/2.0,height/2.0], alpha=0.5)
    fname=file[:file.find(".")]
    plt.savefig(result + fname + ".eps", format="eps", bbox_inches='tight')
    plt.clf()
