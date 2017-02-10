import pickle
import numpy as np
from evaluation import evaluate_ours, evaluate_lin, classifier
#import matplotlib.pyplot as plt
#from matplotlib import cm
#import matplotlib.image as mpimg
import generate_distributions
from generate_distributions import linear_generator
#from test_distribution import particle_generator as particle_generator_t
#from decimal import Decimal
#from adjustText import adjust_text
#from sklearn.metrics import roc_curve, auc, precision_recall_curve
#from visualization_routines import singular_distribution_to_image
#from helper_routines import convolve_and_score

from data import scenes as test_scenes
from data import sets as test_sets
from data import random_sigmas

from sys import argv
scene_number = int(argv[1])
test_scene = test_scenes[int(argv[1])]
scene = test_scene
test_set = test_sets[int(argv[1])]
generate_distributions.set_scene(int(argv[1]))

inds = range(int(argv[2]))

def rho_true(subj, T, test_set, bbox_ls):
    """ Computes the integral of rho_true over a bounding box"""
    res = (20,20)
    x = 0.5*(test_set[subj][0,T] + test_set[subj][2,T])
    y = 0.5*(test_set[subj][1,T] + test_set[subj][3,T])

    x_min = x-test_scene.bbox_width/2
    x_max = x+test_scene.bbox_width/2
    y_min = y-test_scene.bbox_width/2
    y_max = y+test_scene.bbox_width/2
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
    out /= test_scene.bbox_width**2
    return out


def evaluate(gen, i, t_final, N_points):

    return predic, true, rho_arr

def plot_roc(predics, trues, title, axes):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(trues, predics)
    axes.scatter(false_positive_rate, true_positive_rate)
    txts = []
    #for i in range(len(false_positive_rate)):
        #txts.append(axes.text(false_positive_rate[i], true_positive_rate[i], '%.2E' % Decimal(thresholds[i])))
    #adjust_text(txts, ax=axes, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
    axes.set_title(title)
    return auc(false_positive_rate, true_positive_rate)



if __name__ == "__main__":
    import cProfile, pstats, StringIO
    import matplotlib.pyplot as plt
    import time
    from integrate import trap_quad
    import matplotlib.pyplot as plt
    import matplotlib.animation as anim
    import types
    st = time.time()
    print time.time()
    def f(i):
        test_BB_ts = test_set[i]

        from process_data import BB_ts_to_curve
        curve = BB_ts_to_curve( test_BB_ts)

        x_hat = curve[:,1]
        v_hat = (curve[:,10] - curve[:,0])/10
        print "x_hat = " + str(x_hat)
        print "v_hat = " + str(v_hat)
        speed = np.sqrt(np.sum(v_hat**2))
        print "Measured speed / sigma_L = {:f}".format( speed / scene.sigma_L )
        print "sigma_L = {:f}".format( scene.sigma_L)
        k=0
        t_final = min(len(curve[0]), 400)
        N_steps = t_final
        #Domain is actually larger than the domain we care about
        domain = [-scene.width/2, scene.width/2, -scene.height/2, scene.height/2]
        bounds = [scene.width, scene.height]
        #ours = particle_generator(x_hat, v_hat, t_final, N_steps, convolve=False)
        #mine = particle_generator_t(x_hat, v_hat, t_final, N_steps)
        lin = linear_generator(x_hat, v_hat, t_final, N_steps)
        n = 0
        res = (50,60)
        ims = []
        #fig = plt.figure()
        true = np.array([])
        predic = np.array([])

        rho_arr = []

        auc_ours = []
        auc_lin = []
        ppr_ours = np.array([])
        ttr_ours = np.array([])
        ppr_lin = np.array([])
        ttr_lin = np.array([])
        for t in range(N_steps):
            print n
            n += 1
            width = scene.width/40.0
            rand_rho = (np.array([[x_hat[0]], [x_hat[1]]]), [1])
            rand_walk, _ = classifier(bounds, width, rand_rho, t_final/float(N_steps) * n * random_sigmas[scene_number])
            continue
    import time
    for i in inds:
        st = time.time()
        f(i)
        print "TIME: {}".format(time.time()-st)


