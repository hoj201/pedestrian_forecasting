import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sys import argv
import os
name = argv[1]

def fn(paths):
    fs_tr = []
    fs_pr = []

    for path in paths:
        for file in os.listdir(path):
            if "AUC" in file or "DS" in file:
                continue
            undscrs = [i for i, ltr in enumerate(file) if ltr == "_"]
            ag_num = int(file[undscrs[1]+1:undscrs[2]])
            t = int(file[undscrs[3]+1:file.find(".")])
            arr = np.load(path + file)
            fs_pr.append((ag_num, t, arr)) if "pr" in file else fs_tr.append((ag_num, t, arr))

    ag_nums = [x[0] for x in fs_tr]
    ts = sorted(list(set([x[1] for x in fs_tr])))
    num_agents = max(ag_nums)
    aggr_pr = [[] for x in range(len(ts))]
    aggr_tr = [[] for x in range(len(ts))]

    [aggr_pr[ts.index(point[1])].append(point) for point in fs_pr]
    [aggr_tr[ts.index(point[1])].append(point) for point in fs_tr]
    sort_map = lambda x: x[0]
    mappity = lambda x: sorted(x, key=sort_map)
    aggr_pr = map(mappity, aggr_pr)
    aggr_tr = map(mappity, aggr_tr)

    aucs = []
    times = []
    for time in zip(aggr_pr, aggr_tr):
        t = time[0][0][1]
        if len(time[0]) < num_agents: continue
        times.append(t)
        prs = time[0][0][2]
        trs = time[1][0][2]
        for ag_num in range(1, len(time[0])):
            prs = np.concatenate((prs, time[0][ag_num][2]))
            trs = np.concatenate((trs, time[1][ag_num][2]))
        fpr, tpr, thr = roc_curve(trs, prs)
        aucs.append(auc(fpr, tpr))

    plt.title("AUC vs time")
    ax = plt.gca()
    ax.set_ylim([-.1, 1.1])
    ax.set_ylabel("AUC")
    ax.set_xlabel("Frames")
    ax.plot(times, aucs)
    plt.savefig("images/kitani/AUC_vs_t_for_{}.png".format(name))
    plt.show()
    return aucs, times

paths_kit = ["pickles/kitani/{}/".format(x) for x in range(5)]
paths_ours = ["pickles/ours/{}/".format(x) for x in range(5)]
paths_lin = ["pickles/linear/{}/".format(x) for x in range(5)]
our_auc, our_times = fn(paths_ours)
kit_auc, kit_times = fn(paths_kit)
lin_auc, lin_times = fn(paths_lin)

ax = plt.gca()
ax.set_ylim([-.1, 1.1])
ax.set_xlabel("Frames")
ax.set_ylabel("AUC")
plt.plot(our_times, our_auc, ls="solid", c="black", label="Our Algorithm")
plt.plot(kit_times, kit_auc, ls="dashed", c="black", label="Kitani Algorithm")
plt.plot(lin_times, lin_auc, ls="dashdot", c="black", label="Linear Predictor")
plt.legend(loc="lower right")
fig.savefig('images/The Results.eps', format='eps')
plt.show()














