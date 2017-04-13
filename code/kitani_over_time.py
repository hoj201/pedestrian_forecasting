import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sys import argv
import os
#name = argv[1]

def fn(paths):
    fs_tr = []
    fs_pr = []

    for path in paths:
        print path
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
    fprs = []
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
        def whythefuck(ft):
            f,t = ft
            y =  (1 + t - f) / 2.0
            return (1-y, y)
        pts = map(whythefuck, zip(fpr, tpr))
        l2 = lambda x: np.sqrt((x[0][0]-x[1][0])**2 + (x[0][1]-x[1][1])**2)
        dists = map(l2, zip(pts, zip(fpr, tpr)))
        ind = dists.index(min(dists))
        the_tpr_1 = (fpr[ind], tpr[ind])
        dists[dists.index(min(dists))] = 1000000000
        ind = dists.index(min(dists))
        the_tpr_2 = (fpr[ind], tpr[ind])
        m = (the_tpr_1[1] - the_tpr_2[1]) / (the_tpr_1[0] - the_tpr_1[1])
        the_tpr = 1-(the_tpr_1[1] - m * the_tpr_1[0] - 1) / (-1 * (m + 1))
        aucs.append(auc(fpr, tpr))

        thr = list(thr)
        tprr = tpr[thr.index(sorted(thr)[-1 * len(thr)/2])]

        #fprs.append(np.average(tpr))

        fprs.append(the_tpr)

    plt.title("AUC vs time")
    ax = plt.gca()
    ax.set_ylim([-.1, 1.1])
    ax.set_ylabel("AUC")
    ax.set_xlabel("Frames")
    ax.plot(times, aucs)
    #plt.savefig("images/kitani/AUC_vs_t_for_{}.png".format(name))
    #plt.show()
    return np.array(aucs), np.array(fprs), np.array(times)
 
from json_help import read_json
order = read_json("scene_order.json")['order']
params = read_json("params.json")

rng = [0, 1, 2, 3]
n_fold = [0, 1]
sm = lambda x, y: x + y
paths_kit = reduce(sm, [["pickles/kitani/{}/{}/".format(order[x], j) for j in n_fold] for x in rng])
paths_ours = reduce(sm, [["pickles/ours/{}/{}/".format(order[x], j) for j in n_fold] for x in rng])
paths_rand = reduce(sm, [["pickles/rand/{}/{}/".format(order[x], j) for j in n_fold] for x in rng])

our_auc, our_fpr, our_times = fn(paths_ours)
kit_auc, kit_fpr, kit_times = fn(paths_kit)
#lin_auc, lin_times = fn(paths_lin)
rand_auc, rand_fpr, rand_times = fn(paths_rand)
plt.clf()
ax = plt.gca()
ax.set_ylim([-.1, 1.1])
ax.set_xlabel("time")
ax.set_ylabel("AUC")
plt.plot(our_times/30.0, our_auc, ls="solid", c="black", label="Our Algorithm")
plt.plot(kit_times/30.0, kit_auc, ls="dashed", c="black", label="Kitani Algorithm")
#plt.plot(lin_times, lin_auc, ls="dashdot", c="black", label="Linear Predictor")
plt.plot(rand_times/30.0, rand_auc, ls="dashdot", c="black", label="Random Walk")
plt.legend(loc="lower right")
plt.savefig('images/The Results.eps', format='eps')
plt.show()

plt.clf()
ax = plt.gca()
ax.set_ylim([-.1, 1.1])
ax.set_xlabel("times")
ax.set_ylabel("FPR")
plt.plot(our_times/30.0, our_fpr, ls="solid", c="black", label="Our Algorithm")
plt.plot(kit_times/30.0, kit_fpr, ls="dashed", c="black", label="Kitani Algorithm")
#plt.plot(lin_times, lin_auc, ls="dashdot", c="black", label="Linear Predictor")
plt.plot(rand_times/30.0, rand_fpr, ls="dashdot", c="black", label="Random Walk")
plt.legend(loc="lower right")
plt.savefig('images/The FPR Results.eps', format='eps')
plt.show()



#from data import all_data

assert False
plt.clf()
ax = plt.gca()
print "starting evil"

def cold_blooded_murder(paths):
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
    print "begin second for"
    all_data = []
    fs_tr_sorted = []
    for i in fs_pr:
        for j in fs_tr:
            if j[0] == i[0] and j[1] == i[1]:
                all_data.append(j)
                continue
    pr_arrs = [x[2] for x in fs_pr]
    tr_arrs = [x[2] for x in fs_tr_sorted]
    print "beginning sum"
    sm = lambda x, y: np.concatenate((x, y))
    comp_pr = np.array([])
    comp_tr = np.array([])
    for ind, x in enumerate(pr_arrs):
        comp_pr = np.concatenate( (comp_pr, x))
    for ind, x in enumerate(tr_arrs):
        comp_tr = np.concatenate((comp_tr, x))
    print "begin roc"
    fpr, tpr, thr = roc_curve(comp_tr, comp_pr)
    return fpr, tpr

print "begin ours"
fpr_ours, tpr_ours = cold_blooded_murder(paths_ours)
print "begin kit"
fpr_kit, tpr_kit = cold_blooded_murder(paths_kit)
#lin_auc, lin_times = fn(paths_lin)
print "begin kit"
fpr_rand, tpr_rand = cold_blooded_murder(paths_rand)


print "begin plot"
ax.set_ylim([-.1, 1.1])
ax.set_xlim([-.1, 1.1])
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
plt.plot(fpr_ours, tpr_ours, ls="solid", c="black", label="Our Algorithm")
plt.plot(fpr_kit, tpr_kit, ls="dashed", c="black", label="Kitani Algorithm")
#plt.plot(lin_times, lin_auc, ls="dashdot", c="black", label="Linear Predictor")
plt.plot(fpr_rand, tpr_rand, ls="dashdot", c="black", label="Random Walk")
plt.legend(loc="lower right")
plt.savefig('images/The ROC Results.eps', format='eps')
print "end plot"
plt.show()







