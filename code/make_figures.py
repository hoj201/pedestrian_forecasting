from sys import argv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from data import sets, scenes
from process_data import BB_ts_to_curve as bbts
import json
file = argv[1]
with open(file) as f:
    st = f.read()
json_acceptable_string = st.replace("'", "\"")
dic = json.loads(json_acceptable_string)
scene_number = dic['scene_number']
scene = scenes[scene_number]
sett = sets[scene_number]
width = dic['width']
height = dic['height']
folders = dic['folders']
labels = dic['labels']
reference = dic['reference']
agent = dic['agent']
times = dic['times']

fnames = ["pr_agent_{}_time_{}.npy".format(agent, t) for t in times]

methods = [[np.load(x + f) for f in fnames] for x in folders]
mx = np.amax(np.array([[np.amax(x) for x in row] for row in methods]))
nummth = len(methods)
numt = len(times)
curve = bbts(sett[agent])
begin = curve[:, 0]
end = curve[:, -1]

for parad in range(nummth):
    for time in range(numt):
        fig,ax = plt.subplots(1)
        fig.subplots_adjust(left=-scene.width/2,right=scene.width/2,bottom=-scene.height/2,top=scene.height/2)
        ax.set_ylim([-scene.height/2,scene.height/2])
        ax.set_xlim([-scene.width/2, scene.width/2])
        ax.axis("off")


        ax.imshow(methods[parad][time].reshape(width, height).transpose(), origin="lower", extent=[-scene.width/2,scene.width/2,-scene.height/2,scene.height/2], cmap="viridis")
        ax.imshow(mpimg.imread(reference), extent=[-scene.width/2, scene.width/2,-scene.height/2,scene.height/2], alpha=0.5)
        ax.plot(curve[0, :times[time]], curve[1, :times[time]], c="white", alpha=dic['linealpha'])
        ax.scatter(begin[0], begin[1], marker="o", c="white", edgecolors="none", s=dic['markersize'])
        ax.scatter(end[0], end[1], marker="x", c="white", edgecolors="none", s=dic['markersize'])
        ax.scatter(curve[0, times[time]], curve[1, times[time]], marker="D", c="white",  edgecolors="none", s=dic['markersize'])
        plt.savefig(dic['resultsfolder'] + "{}_scene_{}_agent_{}_time_{}.eps".format(labels[parad], scene_number, agent, times[time]), format="eps", bbox_inches='tight', pad_inches=0)
#plt.suptitle(dic['title'], size=dic['titlesize'])
#plt.subplots_adjust( wspace=dic['imgspacing'], hspace=dic['imgspacing'])
