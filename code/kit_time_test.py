from data import sets, scenes
import numpy as np
from sys import argv
import os
from process_data import BB_ts_to_curve as bbts
import subprocess
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from visualization_routines import singular_distribution_to_image
from evaluation import evaluate_ours, evaluate_lin
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import json
file = argv[1]
with open("scene_order.json") as f:
    st = f.read()
json_acceptable_string = st.replace("'", "\"")
dic = json.loads(json_acceptable_string)


scene_number = int(argv[1])
dims = dic['dims']
width = dims[scene_number][0]
height = dims[scene_number][1]

inds = range(int(argv[2]))

scene = scenes[scene_number]
set = sets[scene_number]
def coord_change(begin, end):
    begin[0] += scene.width/2.0
    end[0] += scene.width/2.0
    begin[1] = (-1 * begin[1] + scene.height/2)
    end[1] = (-1 * end[1] + scene.height/2)
    begin *= width
    end *= width
    return begin, end



"""
workflow:
set terminal_points
run script
copy frames
analyze frames
save result
repeat for all agents.
"""
#curve = bbts(agent)
def f(i):
    agent = sets[scene_number][i]
    print agent.shape
    curve = bbts(agent)
    begin = curve[:, 0]
    end = curve[:, -1]
    print begin
    print end
    begin, end = coord_change(begin, end)
    print begin
    print end
    with open("kitani/oc_demo/walk_terminal_pts.txt", "w") as f:
        f.write("{} {}\n{} {}".format(int(begin[0]), int(begin[1]), int(end[0]), int(end[1])))
    process = subprocess.Popen(["./kitani/theirs", "oc_demo"], stdout=subprocess.PIPE)
    output, err = process.communicate()

import time
for i in inds:
    st = time.time()
    f(i)
    print "TIME: {}".format(time.time() - st)
