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



scene_number = int(argv[1])
width = int(argv[2])
height = int(argv[3])
sf = min(width, height)

scene = scenes[scene_number]
set = sets[scene_number]
def coord_change(begin, end):
    begin[0] += scene.width/2.0
    end[0] += scene.width/2.0
    begin[1] = (-1 * begin[1] + scene.height/2)
    end[1] = (-1 * end[1] + scene.height/2)
    begin *= sf
    end *= sf
    return begin, end

def rho_true(subj, T, test_set, bbox_ls):
    """ Computes the integral of rho_true over a bounding box"""
    res = (20,20)
    x = 0.5*(test_set[subj][0,T] + test_set[subj][2,T])
    y = 0.5*(test_set[subj][1,T] + test_set[subj][3,T])

    x_min = x-scene.bbox_width/2
    x_max = x+scene.bbox_width/2
    y_min = y-scene.bbox_width/2
    y_max = y+scene.bbox_width/2
    x_left = bbox_ls[:, 0]
    x_right = bbox_ls[:, 1]
    y_bottom = bbox_ls[:, 2]
    y_top = bbox_ls[:, 3]
    pmax = lambda x,y: x*(x>y)+y*(y>x)
    pmin = lambda x,y: x*(x<y)+y*(y<x)
    x_left = pmax(x_min, x_left)
    x_right = pmin(x_max, x_right)
    y_bottom = pmax(y_min, y_bottom)
    y_top = pmin(y_max, y_top)
    out = (x_right-x_left)*(x_right>x_left)*(y_top-y_bottom)*(y_top>y_bottom)
    out /= scene.bbox_width**2
    return out


"""
workflow:
set terminal_points
run script
copy frames
analyze frames
save result
repeat for all agents.
"""

for j, agent in enumerate(set):
    curve = bbts(agent)
    curve1 = bbts(agent)
    begin = curve[:, 0]
    end = curve[:, -1]
    print begin
    print end
    begin, end = coord_change(begin, end)
    print begin
    print end
    with open("kitani/oc_demo/walk_terminal_pts.txt", "w") as f:
        f.write("{} {}\n{} {}".format(int(begin[0]), int(begin[1]), int(end[0]), int(end[1])))
    process = subprocess.Popen("./kitani/theirs", stdout=subprocess.PIPE)
    output, err = process.communicate()
    print output

    ls = os.listdir("kitani/oc_demo/frames")
    ln = len(ls)
    dic = [0 for x in range(ln)]
    for file in os.listdir("kitani/oc_demo/frames"):
        if file == ".DS_Store":
            continue
        num = int(file[file.index("e")+1:file.index(".")])
        print num
        data = cv2.cv.Load("kitani/oc_demo/frames/" + file)
        data = np.array(data)
        xs = []
        ys = []
        box_width = 1/float(width)
        box_height = 1/float(height)
        #bboxes = []
        for h in range(height):
            for w in range(width):
                xs.append(scene.width*(-1/2.0+ w * box_width + box_width/2))
                ys.append(scene.height*(1/2.0 - h * box_height + box_height/2))
                #bboxes.append([-scene.width/2 + w * box_width, -scene.width/2 + (w+1) * box_width, -scene.height/2 + h * box_height, -scene.height/2 + (h + 1) * box_height])
        rt = lambda x: rho_true(scene_number, int(len(set)/float(ln) * num), set, x)
        rhol = (np.vstack((xs, ys)), data.flatten())
        #plt.imshow(data)
        #plt.show()
        #plt.scatter(xs, ys, c=data.flatten(), cmap="viridis", edgecolors="none")
        #plt.show()
        box_w = scene.width/40
        pr_lin, tr_lin = evaluate_lin([scene.width, scene.height], rhol, rt, box_w, debug_level=0)
        false_pos, true_pos, thresh = roc_curve(tr_lin, pr_lin)
        dic[num] = auc(false_pos, true_pos)
        np.save("pickles/kitani/{}/pr_agent_{}_time_{}".format(scene_number, j, num), pr_lin)
        np.save("pickles/kitani/{}/tr_agent_{}_time_{}".format(scene_number, j, num), tr_lin)
        ctx = int(np.ceil(scene.width/box_w))
        cty = int(np.ceil(scene.height/box_w))
        print ctx
        print cty
        assert False
        plt.title("ROC for agent {} t={}".format(j, num))
        fig, ax = plt.subplots(2)
        ax[0].set_ylim([-.1, 1.1])
        ax[0].set_xlim([-.1, 1.1])
        ax[0].set_ylabel("True Positive Rate")
        ax[0].set_xlabel("False Positive Rate\nAUC={}".format(dic[num]))
        ax[0].plot(false_pos, true_pos)
        ax[1].imshow(pr_lin.reshape(ctx, cty).transpose(), origin="lower", extent=[-scene.width/2,scene.width/2,-scene.height/2,scene.height/2], cmap="viridis")
        ax[1].imshow(mpimg.imread("kitani/oc_demo/walk_birdseye.jpg"), extent=[-scene.width/2, scene.width/2,-scene.height/2,scene.height/2], alpha=0.5)
        plt.plot(curve1[0], curve1[1])
        plt.savefig("images/kitani/{}/AUC_for_agent_{}_t={}.png".format(scene_number, j, num))
        plt.clf()
        plt.close('all')
    np.save("pickles/kitani/{}/AUC_agent_{}".format(scene_number, j), np.array(dic))
    plt.title("AUC for agent {} kitani".format(j))
    fig = plt.figure()
    ax = plt.gca()
    ax.set_ylim([-.1, 1.1])
    ax.set_ylabel("AUC")
    ax.set_xlabel("Frames")
    ax.plot([x for x in range(ln)], dic)
    plt.savefig("images/kitani/AUC_for_agent_{}.png".format(j))
    plt.clf()
    [os.remove("kitani/oc_demo/frames/" + x) for x in os.listdir("kitani/oc_demo/frames")]

