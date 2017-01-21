import pickle
import numpy as np
from evaluation import evaluate_plane
import matplotlib.pyplot as plt
from matplotlib import cm
from generate_distributions import particle_generator, lin_generator
from test_distribution import particle_generator as particle_generator_t
from decimal import Decimal
from adjustText import adjust_text
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from visualization_routines import singular_distribution_to_image

from data import scenes as test_scenes
from data import sets as test_sets


from sys import argv
if len(argv) < 2:
    qscene_number = 0
    test_scene = test_scenes[0]
    scene = test_scene
    test_set = test_sets[0]
else:
    scene_number = int(argv[1])
    test_scene = test_scenes[int(argv[1])]
    scene = test_scene
    test_set = test_sets[int(argv[1])]



def rho_true(subj, T, test_set, bbox_ls):
    """ Computes the integral of rho_true over a bounding box"""
    res = (20,20)
    x = 0.5*(test_set[subj][0,T] + test_set[subj][2,T])
    y = 0.5*(test_set[subj][1,T] + test_set[subj][3,T])
    x_min = x-test_scene.bbox_width/2
    x_max = x+test_scene.bbox_width/2
    y_min = y-test_scene.bbox_width/2
    y_max = y+test_scene.bbox_width/2
    bbox_npy = np.array(bbox_ls)
    x_width_arr = bbox_npy[:,0,0]
    x_pos_arr = bbox_npy[:,1,0]
    y_width_arr = bbox_npy[:,0,1]
    y_pos_arr = bbox_npy[:,1,1]
    pmax = lambda x,y: x*(x>y)+y*(y>x)
    pmin = lambda x,y: x*(x<y)+y*(y<x)
    x_left = pmax(x_min, x_pos_arr - x_width_arr/2.0)
    x_right = pmin(x_max, x_pos_arr + x_width_arr/2.0)
    y_bottom = pmax(y_min, y_pos_arr - y_width_arr/2.0)
    y_top = pmin(y_max, y_pos_arr + y_width_arr/2.0)
    out = (x_right-x_left)*(x_right>x_left)*(y_top-y_bottom)*(y_top>y_bottom)
    out /= test_scene.bbox_width**2
    return out


def evaluate(gen, i, t_final, N_points):
    n = 0
    from visualization_routines import singular_distribution_to_image
    res = (50,60)
    ims = []
    #fig = plt.figure()
    true = np.array([])
    predic = np.array([])

    rho_arr = []

    for x_arr, w_arr in gen:
        if n%1==0:
            rho_arr.append((x_arr, w_arr))
            w_arr /= np.sum(w_arr) if  np.sum(w_arr) > 0 else 1
            #whr = np.where(w_arr > 0)[0] 
            #x_arr = x_arr.transpose()[whr].transpose()
            #w_arr = w_arr[whr]
            #X,Y,Z = singular_distribution_to_image( x_arr, w_arr, domain, res=res)
            #plt.contourf(X,Y,Z, 30, cmap='viridis')

            print "{} steps into agent {}".format(n, i)
            bounds = [test_scene.width, test_scene.height]
            rho = (x_arr, w_arr)
            rt = lambda x: rho_true(i, int(t_final/float(N_points) * n), test_set, x)
            width = test_scene.width/100
            pr, tr = evaluate_plane(bounds, rho, rt, width, debug_level=0)

            if len(true) > 0:
                true = np.vstack((true, tr))
                predic = np.vstack((predic, pr))
            else:
                true = np.array([tr])
                predic = np.array([pr])
        n += 1

    return predic, true, rho_arr

def plot_roc(predics, trues, title, axes, f):
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

    tau_arr = np.array([10**(-x) for x in range(2, 5)])
    tau_arr = np.exp( -np.log(10) * np.linspace(0, 6, 8))
    test_set = test_set[0:6]

    xs_l = np.zeros(0)

    ys_l = np.zeros(0)

    zs_prec_ours = np.zeros(0)
    zs_rec_ours = np.zeros(0)

    zs_prec_mine = np.zeros(0)
    zs_rec_mine = np.zeros(0)

    zs_prec_lin = np.zeros(0)
    zs_rec_lin = np.zeros(0)

    prec_arr_ours = np.zeros(len(tau_arr))
    rec_arr_ours = np.zeros(len(tau_arr))

    prec_arr_lin = np.zeros(len(tau_arr))
    rec_arr_lin = np.zeros(len(tau_arr))


    res_ours = np.zeros(3)
    res_mine = np.zeros(3)
    res_lin = np.zeros(3)
    f = open("results_coupa.txt", "w")

    trueours = []
    predicours = []

    truemine = []
    predicmine = []

    truelin = []
    prediclin = []
    f_lin = open("results/linear.txt", "w")
    f_ours = open("results/ours.txt", "w")

    for i in range(0, len(test_set)):
	print len(test_set)
        test_BB_ts = test_set[i]

        from process_data import BB_ts_to_curve
        curve = BB_ts_to_curve( test_BB_ts)

        x_hat = curve[:,1]
        v_hat = (curve[:,100] - curve[:,0])/100
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
        ours = particle_generator(x_hat, v_hat, t_final, N_steps)
        mine = particle_generator_t(x_hat, v_hat, t_final, N_steps)
        lin = lin_generator(x_hat, v_hat, t_final, N_steps)

        predico, trueo, rho_arro = evaluate(ours, i, t_final, N_steps)
        #predicm, truem = evaluate(mine, i, t_final, N_steps)
        #predicmine.append(predicm)
        #truemine.append(truem)
        auc_ours = []
        auc_lin = []
        predicl, truel, rho_arrl = evaluate(lin, i, t_final, N_steps)
        if len(predicours) == 0:
            predicours = predico
            trueours = trueo
            prediclin = predicl
            truelin = truel
        else:
            ind = min(len(predicours), len(predico))
            predicours = np.hstack((predicours[:ind], predico[:ind]))
            trueours = np.hstack((trueours[:ind], trueo[:ind]))
            prediclin = np.hstack((prediclin[:ind], predicl[:ind]))
            truelin = np.hstack((truelin[:ind], truel[:ind]))

        for k in range(len(predico)):

            f, axarr = plt.subplots(2, 2)
            for ax in axarr[0]:
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_xlim([-.1, 1.2])
                ax.set_ylim([-.1, 1.2])




            auco = plot_roc(predico[k], trueo[k], "Our Algorithm for agent {}, t={}".format(i, int(t_final/float(N_steps) * k)), axarr[0][0], f_ours)
            auc_ours.append(auco)
            aucl = plot_roc(predicl[k], truel[k], "Linear Predictor for agent {}, t={}".format(i, int(t_final/float(N_steps) * k)), axarr[0][1], f_lin)
            auc_lin.append(aucl)
            X,Y,Z = singular_distribution_to_image(
                rho_arro[k][0], rho_arro[k][1], domain, res= (100,100))
            #Z = Z > 1E-3
            im = axarr[1][0].pcolormesh(X,Y,Z, cmap='viridis')
            axarr[1][0].set_xlabel("AUC is {}".format(auco))

            bounds = [[-scene.width/2, scene.width/2], [-scene.height/2, scene.height/2]]
            bounds2 = [[-.1, 1.2], [-.1, 1.2]]
            axarr[1][0].set_xlim(bounds[0])
            axarr[1][0].set_ylim(bounds[1])
            axarr[1][1].set_xlim(bounds[0])
            axarr[1][1].set_ylim(bounds[1])


            x = curve[0][int(t_final/float(N_steps) * k )]
            y = curve[1][int(t_final/float(N_steps) * k )]

            xs = [x - test_scene.bbox_width/2.0, x - test_scene.bbox_width/2.0, x + test_scene.bbox_width/2.0, x + test_scene.bbox_width/2.0, x - test_scene.bbox_width/2.0]
            ys = [y - test_scene.bbox_width/2.0, y + test_scene.bbox_width/2.0, y+ test_scene.bbox_width/2.0, y - test_scene.bbox_width/2.0, y - test_scene.bbox_width/2.0]

            axarr[1][0].scatter(x, y)

            axarr[1][0].plot(xs, ys)

            X,Y,Z = singular_distribution_to_image(
                rho_arrl[k][0], rho_arrl[k][1], domain, res= (100,100))
            #Z = Z > 1E-3
            im = axarr[1][1].pcolormesh(X,Y,Z, cmap='viridis')
            axarr[1][1].set_xlabel("AUC is {}".format(aucl))

            axarr[1][1].scatter(x, y)

            axarr[1][1].plot(xs, ys)

            #plt.savefig("images/precision_recall/{}/pr_agent{}_T{}.png".format(scene_number,i, int(t_final/float(N_steps) * k)))
            #plt.show()
            plt.close('all')

        fig = plt.figure()
        ax = plt.gca()
        ax.set_ylim([-.1, 1.2])
        plt.title("AUC for agent {}".format(i))
        ax.plot([x for x in range(len(auc_ours))], auc_ours, label = "Ours")
        ax.plot([x for x in range(len(auc_lin))], auc_lin, label = "Linear")
        ax.set_xlabel('Frames')
        ax.set_ylabel('AUC')
        ax.legend()
        plt.savefig("images/precision_recall/{}/AUC_for_agent_{}.png".format(scene_number, i))
        plt.close('all')




    for t in range(len(predicours)):

        f, axarr = plt.subplots(1, 2)
        for ax in axarr:
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_aspect('equal')
            ax.set_xlim([-.1, 1.2])
            ax.set_ylim([-.1, 1.2])




        auco = plot_roc(predicours[t], trueours[t], "Our algorithm over all agents, scene={}, t={}".format(scene_number, int(t_final/float(N_steps) * t)), axarr[0], f_ours)
        auc
        aucl = plot_roc(prediclin[t], truelin[t], "Linear Predictor over all agents, scene={},  t={}".format(scene_number, int(t_final/float(N_steps) * t)), axarr[1], f_lin)

        axarr[0].set_xlabel('False Positive Rate\nAUC={}'.format(auco))
        axarr[1].set_xlabel('False Positive Rate\nAUC={}'.format(aucl))

        plt.savefig("images/precision_recall/{}/Over All Agents_T{}.png".format(scene_number, int(t_final/float(N_steps) * t)))
        #plt.show()
        plt.close('all')

    fig = plt.figure()
    f, axarr = plt.subplots(1, 2)
    for ax in axarr:
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_aspect('equal')
        ax.set_xlim([-.1, 1.2])
        ax.set_ylim([-.1, 1.2])




    auco = plot_roc(predicours.flatten(), trueours.flatten(), "Our algorithm over all agents, time", axarr[0], f_ours)

    aucl = plot_roc(prediclin.flatten(), truelin.flatten(), "Linear Predictor over all agents, time", axarr[1], f_lin)

    axarr[0].set_xlabel('False Positive Rate\nAUC={}'.format(auco))
    axarr[1].set_xlabel('False Positive Rate\nAUC={}'.format(aucl))

    plt.savefig("images/precision_recall/Scene {} Over All Agents_Time.png".format(scene_number, int(t_final/float(N_steps) * t)))
    #plt.show()
    plt.close('all')

    



    #plt.show()

    #ax = plt.gca()
    #ax.set_ylim([-.1, 1.1])
    #plt.plot(xs, our_out[1], label='Our Algorithm',lw=5)
    #plt.plot(xs, mine_out[1], label='Perfect Velocity Predictor')
    #plt.plot(xs, lin_out[1], label='Linear Predictor')
    #plt.title("Recall")
    #legend = plt.legend(loc='lower left', shadow=True)
    #plt.savefig("recall_tau={}.png".format(str(tau).replace(".", ",")))
    #plt.show()

    #plt.clf()

    #plt.plot(xs, our_out[2], label='Our Algorithm',lw=5)
    #plt.plot(xs, mine_out[2], label='Perfect Velocity Predictor')
    #plt.plot(xs, lin_out[2], label='Linear Predictor')
    #legend = plt.legend(loc='lower left', shadow=True)
    #plt.title("Accuracy")
    #plt.savefig("accuracy.png")
    #plt.show()
    print "\a" * 100



#plot 2d surfaces






