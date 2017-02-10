from sys import argv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from data import sets, scenes, all_data
from process_data import BB_ts_to_curve as bbts
from helper_routines import ct
import json
file = argv[1]
with open(file) as f:
    st = f.read()
json_acceptable_string = st.replace("'", "\"")
dic = json.loads(json_acceptable_string)
scene_number = dic['scene_number']
split_index = dic['split_index']
scene = all_data[scene_number][split_index][0]
test_set = all_data[scene_number][split_index][1]
with open("scene_order.json") as f:
    st = f.read()
json_acceptable_string = st.replace("'", "\"")
scene_order = json.loads(json_acceptable_string)
scene_name = scene_order['order'][scene_number]
width, height = ct(scene, scene.width/40.0)
reference = scene_order['folders'][scene_number] + "reference.jpg"
folders = [ x + "{}/{}/".format(scene_name, split_index) for x in dic['folders']]
labels = dic['labels']






agent = dic['agent']
times = dic['times']

fnames = ["pr_agent_{}_time_{}.npy".format(agent, t) for t in times]

methods = [[np.load(x + f) for f in fnames] for x in folders]
mx = np.amax(np.array([[np.amax(x) for x in row] for row in methods]))
nummth = len(methods)
numt = len(times)
curve = bbts(test_set[agent])
begin = curve[:, 0]
end = curve[:, -1]

fig, axarr = plt.subplots(nummth, numt, sharex='col', sharey='row')
for ax in axarr.flatten():
    ax.axis('off')
for ct, ax in enumerate(axarr[0, :]):
    ax.text(0, scene.height/2 + dic['fontspacing'], "t={}".format(times[ct]), ha="center", va="center",  size=dic['fontsize'], color=dic['fontcolor'])
for ct, ax in enumerate(axarr[:, 0]):
    ax.text(-scene.width/2 - dic['fontspacing'], 0, labels[ct], ha="center", va="center", rotation=90,  size=dic['fontsize'], color=dic['fontcolor'])

for parad in range(nummth):
    for time in range(numt):
        tmp = axarr[parad][time]
        tmp.imshow(methods[parad][time].reshape(width, height).transpose(), origin="lower", extent=[-scene.width/2,scene.width/2,-scene.height/2,scene.height/2], cmap="viridis")
        tmp.imshow(mpimg.imread(reference), extent=[-scene.width/2, scene.width/2,-scene.height/2,scene.height/2], alpha=0.5)
        tmp.plot(curve[0, :times[time]], curve[1, :times[time]], c="white")
        tmp.scatter(begin[0], begin[1], marker="o", c="white", edgecolors="none", s=dic['markersize'])
        tmp.scatter(end[0], end[1], marker="x", c="white", edgecolors="none", s=dic['markersize'])
        tmp.scatter(curve[0, times[time]], curve[1, times[time]], marker="D", c="white",  edgecolors="none", s=dic['markersize'])
plt.suptitle(dic['title'], size=dic['titlesize'])
plt.subplots_adjust( wspace=dic['imgspacing'], hspace=dic['imgspacing'])
plt.savefig("{}x{} grid.eps".format(nummth, numt), format="eps")
