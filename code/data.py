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

import json
with open("scene_order.json") as f:
    st = f.read()
_json_acceptable_string = st.replace("'", "\"")
_dic = json.loads(_json_acceptable_string)

all_data = []
_order = _dic['order']
for folder in _order:
    row = []
    for num in os.listdir("scenes/{}/scenes/".format(folder)):
           if num == ".DS_Store":
               continue
           with open("scenes/{}/scenes/{}".format(folder, num),'r') as f, \
                open("scenes/{}/sets/{}".format(folder, num),'r') as g, \
                open("scenes/{}/train_sets/{}".format(folder, num),'r') as h:
                row.append((pickle.load(f), pickle.load(g), pickle.load(h)))
    assert len(row) == 10
    all_data.append(row)
