import pickle
import os
import numpy as np
scenes = []
sets = []
for file in sorted(os.listdir("pickles")):
    if file.endswith("scene.pkl"):
        with open("pickles/" + file, "rb") as f:
            scene = pickle.load(f)
            scene.P_of_c = np.ones_like(scene.P_of_c)/float(len(scene.P_of_c))
            
            scenes.append(scene)

    elif file.endswith("set.pkl"):
        with open("pickles/" + file,'r') as f:
            sets.append(pickle.load(f))
scene = scenes[0]
set = sets[0] #TODO: Probably should not use a python keyword here
