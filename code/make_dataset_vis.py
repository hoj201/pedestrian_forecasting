from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
import pickle
import process_data
from scene import Scene
from sys import argv
import numpy as np
from process_data import BB_ts_to_curve as bbts
scene_number = int(argv[1])

from data import scenes
scene = scenes[scene_number]

from json_help import read_json
dic = read_json("scene_order.json")
folder = dic['folders'][scene_number]
name = dic['order'][scene_number]

print "Initializing a scene from " + folder
BB_ts_list, width, height = process_data.get_BB_ts_list(folder,label="Biker")
bb_ls = [bbts(x) for x in BB_ts_list]
import matplotlib.pyplot as plt
import matplotlib.image as img

ref = img.imread(folder + "/reference.jpg")
plt.imshow(ref, extent=[-scene.width/2, scene.width/2, -scene.height/2, scene.height/2], origin="upper")
[plt.plot(x[0], x[1], c="white", alpha=0.5) for x in bb_ls]
ax = plt.gca()
ax.axis('off')
plt.savefig("images/" + name + "_vis.eps", format="eps", bbox_inches="tight")
