import pickle
import os
scenes = []
sets = []
for file in sorted(os.listdir("pickles")):
    if file.endswith("scene.pkl"):
        with open("pickles/" + file, "rb") as f:
            scenes.append(pickle.load(f))
    elif file.endswith("set.pkl"):
        with open("pickles/" + file,'r') as f:
            sets.append(pickle.load(f))
scene = scenes[0]
set = sets[0]
