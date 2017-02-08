import pickle
import os
import numpy as np
from process_data import BB_ts_to_curve as bbts
from train_random_walk import learn_sigma_RW
scenes = []
sets = []
for file in sorted(os.listdir("scenes")):
    if file.endswith("scene.pkl"):
        with open("scenes/" + file, "rb") as f:
            scene = pickle.load(f)
            scene.P_of_c = np.ones_like(scene.P_of_c)/float(len(scene.P_of_c))
            
            scenes.append(scene)

    elif file.endswith("set.pkl"):
        with open("scenes/" + file,'r') as f:
            sets.append(pickle.load(f))
scene = scenes[0]
set = sets[0] #TODO: Probably should not use a python keyword here


random_sigmas = [learn_sigma_RW(curves) for curves in map(lambda x: map(bbts, x), sets)]
 
