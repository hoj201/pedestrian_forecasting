from sys import argv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from data import sets, scenes, all_data
from process_data import BB_ts_to_curve as bbts
from helper_routines import ct
import json
def mkdir(fname):
    try:
        os.mkdir(fname)
    except:
        pass
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
print folders[0]
mxs = [0 for x in range(len(all_data[scene_number][split_index][1]))]
mns = [1000 for x in range(len(all_data[scene_number][split_index][1]))]
for f in os.listdir(folders[0]):

    if "pr_" in f:
        unds = filter(lambda j: f[j] == "_", range(len(f)))
        agnm = int(f[unds[1]+1:unds[2]])
        mxs[agnm] = max(mxs[agnm], int(f[unds[3]+1:f.index(".")]))
        mns[agnm] = min(mns[agnm], int(f[unds[3]+1:f.index(".")]))


timer = [range(p[0], p[1], 5) for p in zip(mns, mxs)]



fnames = [[np.load(folders[0] + "pr_agent_{}_time_{}.npy".format(agt, t)) for t in timer[agt]] for agt in range(len(all_data[scene_number][split_index][1]))]

for agent_num, agent in enumerate(fnames):
    for time, arr in enumerate(agent):
        times = timer[agent_num]
        curve = bbts(test_set[agent_num])
        begin = curve[:, 0]
        end = curve[:, -1]

        #fig, axarr = plt.subplots(nummth, numt, sharex='col', sharey='row')
        #for ax in axarr.flatten():
        #    ax.axis('off')
        #for ct, ax in enumerate(axarr[0, :]):
        #    ax.text(0, scene.height/2 + dic['fontspacing'], "t={}".format(times[ct]), ha="center", va="center",  size=dic['fontsize'], color=dic['fontcolor'])
        #for ct, ax in enumerate(axarr[:, 0]):
        #    ax.text(-scene.width/2 - dic['fontspacing'], 0, labels[ct], ha="center", va="center", rotation=90,  size=dic['fontsize'], color=dic['fontcolor'])
        if labels[0] == "Kitani":
           convt = lambda t: min(float(t)/400 * len(curve[0]), len(curve[0]) -1)
        else:
            convt = lambda t: t
        plt.clf()
        tmp = plt.gca()
        tmp.axis("off")
        tmp.imshow(arr.reshape(width, height).transpose(), origin="lower", extent=[-scene.width/2,scene.width/2,-scene.height/2,scene.height/2], cmap="viridis")
        tmp.imshow(mpimg.imread(reference), extent=[-scene.width/2, scene.width/2,-scene.height/2,scene.height/2], alpha=0.5)
        tmp.plot(curve[0, : convt(times[time])], curve[1, :convt(times[time])], c="white")
        tmp.scatter(begin[0], begin[1], marker="o", c="white", edgecolors="none", s=dic['markersize'])
        tmp.scatter(end[0], end[1], marker="x", c="white", edgecolors="none", s=dic['markersize'])
        tmp.scatter(curve[0, convt(times[time])], curve[1, convt(times[time])], marker="D", c="white",  edgecolors="none", s=dic['markersize'])
        mkdir("images/{}/".format(scene_name))
        mkdir("images/{}/{}".format(scene_name, labels[0]))
        mkdir("images/{}/{}/{}".format(scene_name, labels[0], split_index))
        mkdir("images/{}/{}/{}/{}".format(scene_name, labels[0], split_index, agent_num))
        print "images/{}/{}/{}/{}/agent_{}_time_{}.eps".format(scene_name, labels[0], split_index, agent_num, agent_num, times[time])

        plt.savefig("images/{}/{}/{}/{}/agent_{}_time_{}.eps".format(scene_name, labels[0], split_index, agent_num, agent_num, times[time]), format="eps", bbox_inches="tight")
        #plt.suptitle(dic['title'], size=dic['titlesize'])
        #plt.subplots_adjust( wspace=dic['imgspacing'], hspace=dic['imgspacing'])
        #plt.savefig("{}x{} grid.eps".format(nummth, numt), format="eps")
