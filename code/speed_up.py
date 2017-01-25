import pickle
import numpy as np
from evaluation import evaluate_ours, evaluate_lin
import matplotlib.pyplot as plt
from matplotlib import cm
import generate_distributions
from generate_distributions import particle_generator
#from test_distribution import particle_generator as particle_generator_t
from decimal import Decimal
from adjustText import adjust_text
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from visualization_routines import singular_distribution_to_image
from helper_routines import convolve_and_score

from data import scenes as test_scenes
from data import sets as test_sets

from sys import argv
if len(argv) < 2:
    scene_number = 0
    test_scene = test_scenes[0]
    scene = test_scene
    test_set = test_sets[0]
    inds = range(len(test_set))
else:
    scene_number = int(argv[1])
    test_scene = test_scenes[int(argv[1])]
    scene = test_scene
    test_set = test_sets[int(argv[1])]
    generate_distributions.set_scene(int(argv[1]))

if len(argv) > 2:
        st = argv[2]
        inds = map(int, st.split(","))

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
        ours = particle_generator(x_hat, v_hat, t_final, N_steps, convolve=False)
        #mine = particle_generator_t(x_hat, v_hat, t_final, N_steps)
        #lin = lin_generator(x_hat, v_hat, t_final, N_steps)

        from itertools import izip

        #gen = izip(ours, lin)
        n = 0
        from visualization_routines import singular_distribution_to_image
        res = (50,60)
        ims = []
        #fig = plt.figure()
        true = np.array([])
        predic = np.array([])

        rho_arr = []

        auc_ours = []
        auc_lin = []
        for ((x_arr_ours, w_arr_ours), (x_arr_lin, w_arr_lin)) in ours:
            if n == 0:
                n += 1
                continue
            if n % 5 == 0:
                #whr = np.where(w_arr > 0)[0]
                #x_arr = x_arr.transpose()[whr].transpose()
                #w_arr = w_arr[whr]
                #X,Y,Z = singular_distribution_to_image( x_arr, w_arr, domain, res=res)
                #plt.contourf(X,Y,Z, 30, cmap='viridis')

                print "{} steps into agent {}".format(n, i)
                bounds = [test_scene.width, test_scene.height]
                rho = (x_arr_ours, w_arr_ours)
                rhol = (x_arr_lin, w_arr_lin)
                rt = lambda x: rho_true(i, int(t_final/float(N_steps) * n), test_set, x)
                width = test_scene.width/40
                pr_ours, tr_ours, bboxes = evaluate_ours(bounds, rho, rhol, rt, width, scene.kappa * t_final/float(N_steps) * n, debug_level=0)

                w_arr_lin /= np.sum(w_arr_lin) if  np.sum(w_arr_lin) > 0 else 1
                #whr = np.where(w_arr > 0)[0]
                #x_arr = x_arr.transpose()[whr].transpose()
                #w_arr = w_arr[whr]
                #X,Y,Z = singular_distribution_to_image( x_arr, w_arr, domain, res=res)
                #plt.contourf(X,Y,Z, 30, cmap='viridis')
                rho = (x_arr_lin, w_arr_lin)
                pr_lin, tr_lin = evaluate_lin(bounds, rhol, rt, width, debug_level=0)

                f, axarr = plt.subplots(2, 2)
                for ax in axarr[0]:
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_xlim([-.1, 1.2])
                    ax.set_ylim([-.1, 1.2])

                auc_ours.append(plot_roc(pr_ours, tr_ours, "Our Algorithm for agent {}, t={}".format(i, int(t_final/float(N_steps) * n)), axarr[0][0]))
                auc_lin.append(plot_roc(pr_lin, tr_lin, "Linear Predictor for agent {}, t={}".format(i, int(t_final/float(N_steps) * n)), axarr[0][1]))


                #X,Y,Z = singular_distribution_to_image(
                #    x_arr_ours, w_arr_ours, domain, res= (100,100))
                ctx = int(np.ceil(bounds[0]/width))
                cty = int(np.ceil(bounds[1]/width))

                X = np.zeros((ctx, cty))
                Y = np.zeros((ctx, cty))

                x_lims = np.linspace(-bounds[0]/2, bounds[0]/2, ctx+1)
                y_lims = np.linspace(-bounds[1]/2, bounds[1]/2, cty+1)

                bboxes = []
                sums = []
                for rarg in range(ctx):
                    for karg in range(cty):
                        X[rarg][karg] = (-1 * bounds[0]/2.0 + width * (rarg + 0.5))
                        Y[rarg][karg] = (-1 * bounds[1]/2.0 + width * (karg + 0.5))
                bboxes = np.array(bboxes)

                Z = pr_ours.reshape((ctx, cty))

                im = axarr[1][0].pcolormesh(X,Y,Z, cmap='viridis')
                axarr[1][0].set_xlabel("AUC is {}".format(auc_ours[-1]))

                bounds = [[-scene.width/2, scene.width/2], [-scene.height/2, scene.height/2]]
                bounds2 = [[-.1, 1.2], [-.1, 1.2]]
                axarr[1][0].set_xlim(bounds[0])
                axarr[1][0].set_ylim(bounds[1])
                axarr[1][1].set_xlim(bounds[0])
                axarr[1][1].set_ylim(bounds[1])


                x = curve[0][int(t_final/float(N_steps) * n )]
                y = curve[1][int(t_final/float(N_steps) * n )]

                xs = [x - test_scene.bbox_width/2.0, x - test_scene.bbox_width/2.0, x + test_scene.bbox_width/2.0, x + test_scene.bbox_width/2.0, x - test_scene.bbox_width/2.0]
                ys = [y - test_scene.bbox_width/2.0, y + test_scene.bbox_width/2.0, y+ test_scene.bbox_width/2.0, y - test_scene.bbox_width/2.0, y - test_scene.bbox_width/2.0]

                axarr[1][0].scatter(x, y)

                axarr[1][0].plot(xs, ys)

                X,Y,Z = singular_distribution_to_image(
                    x_arr_lin, w_arr_lin, domain, res= (100,100))
                #Z = Z > 1E-3
                im = axarr[1][1].pcolormesh(X,Y,Z, cmap='viridis')
                axarr[1][1].set_xlabel("AUC is {}".format(auc_lin[-1]))

                axarr[1][1].scatter(x, y)

                axarr[1][1].plot(xs, ys)

                plt.savefig("images/precision_recall/{}/pr_agent{}_T{}.png".format(scene_number, i, int(t_final/float(N_steps) * n)))
                #plt.show()
                plt.close('all')
            n += 1

        fig = plt.figure()
        ax = plt.gca()
        ax.set_ylim([-.1, 1.2])
        plt.title("AUC for agent {}".format(i))
        ax.plot([x for x in range(len(auc_ours))], auc_ours, label = "Ours")
        ax.plot([x for x in range(len(auc_lin))], auc_lin, label = "Linear")
        ax.set_xlabel('Frames')
        ax.set_ylabel('AUC')
        ax.legend()
        plt.savefig("images/precision_recall/{}/{}AUC_for_agent_{}.png".format(scene_number, scene_number, i))
        plt.close('all')

    from joblib import Parallel, delayed
    Parallel(n_jobs=2)(delayed(f)(x) for x in inds)










