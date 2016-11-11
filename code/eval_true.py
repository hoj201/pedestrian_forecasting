import pickle
import numpy as np
from generate_distributions import make_generator
from evaluation import evaluate_plane
import matplotlib.pyplot as plt

with open('test_scene.pkl','r') as f:
    test_scene = pickle.load(f)

with open('test_set.pkl','r') as f:
    test_set = pickle.load(f)

def rho_true(subj, T, test_set, bbox_ls):
    """ Computes the integral of rho_true over a bounding box"""
    res = (20,20)
    x = 0.5*(test_set[subj][0,T] + test_set[subj][2,T])
    y = 0.5*(test_set[subj][1,T] + test_set[subj][3,T])
    print "Rho TRUE"
    print x
    print y
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
    dt = .1
    Nt = 50

    def get_initial_condition(BB_ts):
            fd_width = 4
            BB0 = BB_ts[:,0]
            BB2 = BB_ts[::,fd_width]
            x = 0.5*(BB0[0]+BB0[2]+BB2[0]+BB2[2]) / fd_width
            y = 0.5*(BB0[1]+BB0[3]+BB2[1]+BB2[3]) / fd_width
            u = 0.5*(BB2[0]-BB0[0]+BB2[2]-BB0[2]) / fd_width
            v = 0.5*(BB2[1]-BB0[1]+BB2[3]-BB0[3]) / fd_width
            return np.array([x, y]), np.array([u, v])

    #Iterate through all agents
    for i in range(len(test_set)):
        #Set up generator, code given to me by you.
        test_BB_ts = test_set[i]
        x,v = get_initial_condition(test_BB_ts[:, 10:])
        speed = np.sqrt(np.sum(v**2))
        print "Measured speed / sigma_v = {:f}".format( speed / test_scene.sigma_v )
        print "sigma_v = {:f}".format( test_scene.sigma_v)
        gen = make_generator(test_scene, x, v, dt, Nt)
        print "starting eval"
        #iterate through all time steps
        for (ct, data) in enumerate(gen):
            #ignore predictions where actual data doesn't exist
            if ct % 10 != 0:
                continue #NOTE  hoj:What??
            #Concatenate all xs, ps for the different classes
            xs = np.array([[0,0]])
            ps = np.array([])
            print "Evaluation for agent {}, time {}".format(i, dt * ct)
            for cl in range(test_scene.num_nl_classes):
                xy, p = data[cl]
                p = p.flatten()
                ps = np.concatenate((ps, p))

                for row in xy:
                    xs = np.concatenate((xs, np.array(zip(row[0], row[1]))))
            #delete placeholder component
            xs = np.delete(xs, 0, 0)
            #Define initial conditions
            rho = (xs, ps)
            tau = 0.01
            lin_term = data[-1]

            def lin(x, y):
                x = x.flatten()
                y = y.flatten()
                return lin_term([x,y])

            resolution = np.array([40, 40])
            #Define rho_true for a given time step etc
            rt = lambda bboxes: rho_true(i, ct, test_set,bboxes)
            bbox = np.array([test_scene.width, test_scene.height])
            #Call evaluate_plane
            res =  evaluate_plane(bbox, rho, rt, tau, lin, resolution)
            plt.show()
            print res


