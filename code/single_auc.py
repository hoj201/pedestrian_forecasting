import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sys import argv
import os
#name = argv[1]

def mkdir(fname):
    try:
        os.mkdir(fname)
    except:
        pass

def fn(parad, name, split, agents):
    path = "pickles/{}/{}/{}/".format(parad, name, split)
    fs_tr = []
    fs_pr = []
    agent_amals_tr = [[] for _ in agents]
    agent_amals_pr = [[] for _ in agents]

    for file in os.listdir(path):
        if "AUC" in file or "DS" in file:
            continue
        undscrs = [i for i, ltr in enumerate(file) if ltr == "_"]
        ag_num = int(file[undscrs[1]+1:undscrs[2]])
        if ag_num not in (agents):
            continue
        t = int(file[undscrs[3]+1:file.find(".")])
        arr = np.load(path + file)
        agent_amals_pr[agents.index(ag_num)].append((t, arr)) if "pr" in file else agent_amals_tr[agents.index(ag_num)].append((t,arr))

    sort_map = lambda x: x[0]
    mappity = lambda x: sorted(x, key=sort_map)
    agent_amals_pr = map(mappity, agent_amals_pr)
    agent_amals_tr = map(mappity, agent_amals_tr)
    aucs_ret = []
    times_ret = []
    for ind, agent in enumerate(agents):
        trs = [x[1] for x in agent_amals_tr[ind]]
        prs = [x[1] for x in agent_amals_pr[ind]]
        times = [x[0] for x in agent_amals_tr[ind]]
        res = [roc_curve(x, y) for x,y in zip(trs, prs)]
        aucs = [auc(x[0], x[1]) for x in res]
        aucs_ret.append(aucs)
        times_ret.append(times)
        #ax = plt.gca()
        #ax.set_ylim([-.1, 1.1])
        #ax.set_ylabel("AUC")
        #ax.set_xlabel("Frames")
        #ax.plot(times, aucs)
        #mkdir("images/{}".format(parad))
        #mkdir("images/{}/{}".format(parad, name))
        #mkdir("images/{}/{}/{}".format(parad, name, split))
        #plt.savefig("images/AUC_{}_{}_split_{}_{}.png".format(parad, name, split, agent))
        #plt.show()
    return aucs_ret, times_ret
 
from json_help import read_json
order = read_json("scene_order.json")['order']

indices = read_json(argv[1])['indices']
for ind in indices:
    fig = plt.figure()
    ax = fig.gca()
    ax.set_ylim([-.1, 1.1])
    ax.set_ylabel("AUC")
    ax.set_xlabel("Frames")
    if len(ind) < 4:
        inds = range(3)
    else:
        inds = ind[3]
    if 2 in inds:
        aucs, times = fn("kitani", ind[0], ind[1], [ind[2]])
        ax.plot(times[0], aucs[0], ls="dashed", c="black", label="Kitani et. al.")
    if 0 in inds:
        aucs, times = fn("ours", ind[0], ind[1], [ind[2]])
        ax.plot(times[0], aucs[0], ls="solid", c="black", label="Our Algorithm")
    if 1 in inds:
        aucs, times = fn("rand", ind[0], ind[1], [ind[2]])
        ax.plot(times[0], aucs[0], ls="dashdot", c="black", label="Our Algorithm")
    plt.legend(loc="lower right")
    plt.savefig("images/AUC_{}_split_{}_{}.png".format(ind[0], ind[1], ind[2]))

#our_auc, our_times = fn(paths_ours)
#rand_auc, rand_times = fn(paths_rand)
#kit_auc, kit_times = fn(paths_kit)
#plt.clf()
#ax = plt.gca()
#ax.set_ylim([-.1, 1.1])
#ax.set_xlabel("Frames")
#ax.set_ylabel("AUC")
#plt.plot(our_times, our_auc, ls="solid", c="black", label="Our Algorithm")
#plt.plot(rand_times, rand_auc, ls="dashdot", c="black", label="Random Walk")
#plt.plot(kit_times, kit_auc, ls="dashed", c="black", label="Random Walk")
#plt.legend(loc="lower right")
#plt.savefig('images/The Results.eps', format='eps')
#plt.show()
#






