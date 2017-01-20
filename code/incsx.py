import pickle
from sys import argv
from scene import Scene

file = argv[1]

with open(file, "rb") as f:
    scene = pickle.load(f)

scene.sigma_x *= 10

with open(file, "w") as f:
    pickle.dump( scene, f)
