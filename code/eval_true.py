import pickle
import numpy as np
from evaluation import evaluate_plane
import matplotlib.pyplot as plt
from matplotlib import cm
from generate_distributions import particle_generator, lin_generator
from test_distribution import particle_generator as particle_generator_t

from data import scene as test_scene
from data import set as test_set



def rho_true(subj, T, test_set, bbox_ls):
    """ Computes the integral of rho_true over a bounding box"""
    res = (20,20)
    x = 0.5*(test_set[subj][0,T] + test_set[subj][2,T])
    y = 0.5*(test_set[subj][1,T] + test_set[subj][3,T])
    plt.scatter(x, y, s=60, color="grey")
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


def evaluate(gen, i, t_final, N_points, tau):
    n = 0
    from visualization_routines import singular_distribution_to_image
    res = (50,60)
    ims = []
    #fig = plt.figure()
    results = []
    for x_arr, w_arr in gen:
        if n%5==0:
            whr = np.where(w_arr > 0)[0]
            x_arr = x_arr.transpose()[whr].transpose()
            w_arr = w_arr[whr]
            #X,Y,Z = singular_distribution_to_image( x_arr, w_arr, domain, res=res)
            #plt.contourf(X,Y,Z, 30, cmap='viridis')

            print "{} steps into agent {}".format(n, i)
            bounds = [test_scene.width, test_scene.height]
            rho = (x_arr, w_arr)
            rt = lambda x: rho_true(i, int(t_final/float(N_points) * n), test_set, x)
            width = test_scene.bbox_width/3
            results.append(evaluate_plane(bounds, rho, rt, tau, width, debug_level=0))
        n += 1

    ret = np.zeros(3)
    for result in results:
        ret += np.array(result)

    ret /= len(results)

    return list(ret), results



if __name__ == "__main__":
    import cProfile, pstats, StringIO
    import matplotlib.pyplot as plt
    import time
    from integrate import trap_quad
    import matplotlib.pyplot as plt
    import matplotlib.animation as anim
    import types
    with open('test_set.pkl', 'rs') as f:
        test_set = pickle.load(f)
    with open("test_scene.pkl", "rb") as f:
        scene = pickle.load(f)

    tau_arr = np.array([10**(-x) for x in range(2, 4)])
    test_set = test_set[0:1]

    xs_l = np.zeros(0)

    ys_l = np.zeros(0)

    zs_prec_ours = np.zeros(0)
    zs_rec_ours = np.zeros(0)

    zs_prec_mine = np.zeros(0)
    zs_rec_mine = np.zeros(0)

    zs_prec_lin = np.zeros(0)
    zs_rec_lin = np.zeros(0)


    for (ind, tau) in enumerate(tau_arr):
        print """

======================================
BEGINNING TAU = {}
======================================

        """.format(tau)
        res_ours = np.zeros(3)
        res_mine = np.zeros(3)
        res_lin = np.zeros(3)
        f = open("results_coupa.txt", "w")

        ours_results = []
        mine_results = []
        linear_results = []

        for i in range(0, len(test_set)):
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
            N_steps = 40
            t_final = len(curve[0])
            #Domain is actually larger than the domain we care about
            domain = [-scene.width, scene.width, -scene.height, scene.height]

            ours = particle_generator(x_hat, v_hat, t_final, N_steps)
            mine = particle_generator_t(x_hat, v_hat, t_final, N_steps)
            lin = lin_generator(x_hat, v_hat, t_final, N_steps)
            print "The following are (precision, recall, accuracy) triples:"
            ours, reso = evaluate(ours, i, t_final, N_steps, tau)
            ours_results.append(reso)
            st = "Our algorithm for agent {}: {}".format(i, ours)
            print st
            f.write(st + "\n")
            mine, resm = evaluate(mine, i, t_final, N_steps, tau)
            mine_results.append(resm)
            st = "My test for agent {}: {}".format(i, mine)
            print st
            f.write(st + "\n")
            lin, resl = evaluate(lin, i, t_final, N_steps, tau)
            linear_results.append(resl)
            st = "Linear for agent {}: {}".format(i, lin)
            print st
            f.write(st + "\n")

            res_ours += np.array(ours)
            res_mine += np.array(mine)
            res_lin += np.array(lin)

        res_ours /= len(test_set)
        st = "Total average accuracy for ours {}".format(list(res_ours))
        print st
        f.write(st + "\n")
        res_mine /= len(test_set)
        st = "Total average accuracy for mine {}".format(list(res_mine))
        print st
        f.write(st + "\n")
        res_lin /= len(test_set)
        st = "Total average accuracy for lin {}".format(list(res_lin))
        print st
        f.write(st + "\n")

        #do analysis w.r.t. time



        num_samples = min((len(ours_results[0]), len(mine_results[0]), len(linear_results[0])))
        our_out = np.zeros((3, num_samples))
        mine_out = np.zeros((3, num_samples))
        lin_out = np.zeros((3, num_samples))

        for agent in range(len(ours_results)):
            our_out += np.array(ours_results[agent]).transpose()[:, 0:num_samples]
            mine_out += np.array(mine_results[agent]).transpose()[:, 0:num_samples]
            lin_out += np.array(linear_results[agent]).transpose()[:, 0:num_samples]
        our_out /= len(ours_results)
        mine_out /= len(ours_results)
        lin_out /= len(ours_results)


        plt.clf()
        ax = plt.gca()
        ax.set_ylim([-.1, 1.1])

        xs = range(0, num_samples * 5, 5)

        if len(xs_l) == 0:
            shape = (len(tau_arr), num_samples)
            xs_l = np.zeros(shape)

            ys_l = np.zeros(shape)

            zs_prec_ours = np.zeros(shape)
            zs_rec_ours = np.zeros(shape)

            zs_prec_mine = np.zeros(shape)
            zs_rec_mine = np.zeros(shape)

            zs_prec_lin = np.zeros(shape)
            zs_rec_lin = np.zeros(shape)
        xs_l[ind, :] = xs

        ys_l[ind, :] = np.ones(num_samples) * tau

        zs_prec_ours[ind, :] = our_out[0]
        zs_rec_ours[ind, :] = our_out[1]

        zs_prec_mine[ind, :] = mine_out[0]
        zs_rec_mine[ind, :] = mine_out[1]

        zs_prec_lin[ind, :] = lin_out[0]
        zs_rec_lin[ind, :] = lin_out[1]
        
        ax.set_ylim([-.1, 1.1])
        plt.plot(xs, our_out[0], label='Our Algorithm', lw=5)
        plt.plot(xs, mine_out[0], label='Perfect Velocity Predictor')
        plt.plot(xs, lin_out[0], label='Linear Predictor')
        plt.title("Precision")
        legend = plt.legend(loc='lower left', shadow=True)
        plt.savefig("precision_tau={}.png".format(str(tau).replace(".", ",")))
        plt.show()

        plt.clf()
        ax = plt.gca()
        ax.set_ylim([-.1, 1.1])
        plt.plot(xs, our_out[1], label='Our Algorithm',lw=5)
        plt.plot(xs, mine_out[1], label='Perfect Velocity Predictor')
        plt.plot(xs, lin_out[1], label='Linear Predictor')
        plt.title("Recall")
        legend = plt.legend(loc='lower left', shadow=True)
        plt.savefig("recall_tau={}.png".format(str(tau).replace(".", ",")))
        plt.show()

        plt.clf()

        #plt.plot(xs, our_out[2], label='Our Algorithm',lw=5)
        #plt.plot(xs, mine_out[2], label='Perfect Velocity Predictor')
        #plt.plot(xs, lin_out[2], label='Linear Predictor')
        #legend = plt.legend(loc='lower left', shadow=True)
        #plt.title("Accuracy")
        #plt.savefig("accuracy.png")
        #plt.show()



        f.close()

    #plot 3d surfaces
    fig = plt.figure()
    plt.title("Our Algorithm")
    ax = fig.gca(projection='3d')
    xLabel = ax.set_xlabel('Precision', linespacing=3.2)
    yLabel = ax.set_ylabel('Tau', linespacing=3.1)
    zLabel = ax.set_zlabel('Recall', linespacing=3.4)
    surf = ax.plot_surface(zs_prec_ours, ys_l, zs_rec_ours, rstride=1, cstride=1, cmap=cm.viridis,
                       linewidth=0, antialiased=False)
    plt.show()

    fig = plt.figure()
    plt.title("My Algorithm")
    ax = fig.gca(projection='3d')
    xLabel = ax.set_xlabel('Precision', linespacing=3.2)
    yLabel = ax.set_ylabel('Tau', linespacing=3.1)
    zLabel = ax.set_zlabel('Recall', linespacing=3.4)
    surf = ax.plot_surface(zs_prec_mine, ys_l, zs_rec_mine, rstride=1, cstride=1, cmap=cm.viridis,
                       linewidth=0, antialiased=False)
    plt.show()

    fig = plt.figure()
    plt.title("Lin Algorithm")
    ax = fig.gca(projection='3d')
    xLabel = ax.set_xlabel('Precision', linespacing=3.2)
    yLabel = ax.set_ylabel('Tau', linespacing=3.1)
    zLabel = ax.set_zlabel('Recall', linespacing=3.4)
    surf = ax.plot_surface(zs_prec_lin, ys_l, zs_rec_lin, rstride=1, cstride=1, cmap=cm.viridis,
                       linewidth=0, antialiased=False)
    plt.show()
