import pickle
import numpy as np
from evaluation import evaluate_plane
import matplotlib.pyplot as plt
from generate_distributions import particle_generator
from test_distribution import particle_generator as particle_generator_t

from data import scene as test_scene
from data import set as test_set



def rho_true(subj, T, test_set, bbox_ls):
    """ Computes the integral of rho_true over a bounding box"""
    res = (20,20)
    x = 0.5*(test_set[subj][0,T] + test_set[subj][2,T])
    y = 0.5*(test_set[subj][1,T] + test_set[subj][3,T])
    print "Rho TRUE"
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

    for i in range(0, len(test_set)):
        test_BB_ts = test_set[i]

        from process_data import BB_ts_to_curve
        curve = BB_ts_to_curve( test_BB_ts)
        x_hat = curve[:,5]
        v_hat = (curve[:,100] - curve[:,0])/100
        v_hat /= np.sqrt(v_hat[0] **2 + v_hat[1] ** 2)
        v_hat *= test_scene.sigma_v * 0.5
        print curve[:, 10]
        print curve[:, 0]
        print "x_hat = " + str(x_hat)
        print "v_hat = " + str(v_hat)
        speed = np.sqrt(np.sum(v_hat**2))
        print "Measured speed / sigma_L = {:f}".format( speed / scene.sigma_L )
        print "sigma_L = {:f}".format( scene.sigma_L)
        k=0
        N_steps = 120
        t_final = 120
        #Domain is actually larger than the domain we care about
        domain = [-scene.width, scene.width, -scene.height, scene.height]

        gen = particle_generator_t(x_hat, v_hat, t_final, N_steps)
        n = 0
        from visualization_routines import singular_distribution_to_image
        res = (50,60)
        ims = []
        fig = plt.figure()
        result = []
        for x_arr, w_arr in gen:
            if n%5==0:
                w_arr /= np.sum(w_arr)
                print "{} steps processed for agent {}.".format(n, i)
                #fig = plt.figure()
                X,Y,Z = singular_distribution_to_image(
                        x_arr, w_arr, domain, res=res)
                im = plt.pcolormesh(X,Y,Z, cmap='viridis')
                t_step = int(n/5)

                #################################################################
                ## Bug fix for Quad Contour set not having attribute 'set_visible'
                #def setvisible(self,vis):
                #    for c in self.collections: c.set_visible(vis)
                im.axes = plt.gca()
                im.figure=fig
                ####################################################################
                plt.plot(curve[0], curve[1])
                ims.append([im])

                bounds = [test_scene.width, test_scene.height]
                rho = (np.transpose(x_arr), w_arr)
                width = test_scene.bbox_width/3

                tau = 0

                rt = lambda x: rho_true(i, n, test_set, x)

                result.append(evaluate_plane(bounds, rho, rt, tau, width, debug=True))
            n += 1

        print result
        #plt.show()
