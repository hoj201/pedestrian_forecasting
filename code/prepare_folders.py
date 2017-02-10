import json
import os
from shutil import copy as cp

def mkdir(fname):
    try:
        os.mkdir(fname)
    except:
        pass

from json_help import read_json
dic = read_json("scene_order.json")
order = dic['order']
params = read_json("params.json")

for name in order:
    mkdir("kitani/{}".format(name))
    for ind in range(params['nfold']):
        mkdir("kitani/{}/{}".format(name, ind))
        mkdir("kitani/{}/{}/output".format(name, ind))
        mkdir("kitani/{}/{}/frames".format(name, ind))
        cp("kitani/{}_feat.xml".format(name), "kitani/{}/{}/walk_feature_maps.xml".format(name, ind))
        cp("kitani/{}_topdown.jpg".format(name), "kitani/{}/{}/walk_birdseye.jpg".format(name, ind))
        cp("kitani/{}_reward.txt".format(name), "kitani/{}/{}/walk_reward_weights.txt".format(name, ind))
        
