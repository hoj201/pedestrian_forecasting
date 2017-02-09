import json
import os
from shutil import copy as cp

def mkdir(fname):
    try:
        os.mkdir(fname)
    except:
        pass

with open("scene_order.json") as f:
    st = f.read()
json_acceptable_string = st.replace("'", "\"")
dic = json.loads(json_acceptable_string)
order = dic['order']

for name in order:
    mkdir("kitani/{}".format(name))
    for ind in range(10):
        mkdir("kitani/{}/{}".format(name, ind))
        mkdir("kitani/{}/{}/output".format(name, ind))
        mkdir("kitani/{}/{}/frames".format(name, ind))
        cp("kitani/{}_feat.xml".format(name), "kitani/{}/{}/walk_feature_maps.xml".format(name, ind))
        cp("kitani/{}_topdown.jpg".format(name), "kitani/{}/{}/walk_birdseye.jpg".format(name, ind))
        cp("kitani/{}_reward.txt".format(name), "kitani/{}/{}/walk_reward_weights.txt".format(name, ind))
        
