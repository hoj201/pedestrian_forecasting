import pickle
import numpy as np
from generate_distributions import make_generator
from evaluation import evaluate_plane
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as anim
    import matplotlib.cm as cm
    import time
    FFMpegWriter = anim.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=10, metadata=metadata)
    img=mpimg.imread('background.jpg')
    dt = 1
    Nt = 100
    fig = plt.figure()
    dat = plt.plot([], [], "k-o")
    print "Aspect ratio:"
    print test_scene.width/test_scene.height
    cmap = cm.ScalarMappable(cmap="viridis")

    def get_initial_condition(BB_ts):
            fd_width = 4
            BB0 = BB_ts[:,0]
            BB2 = BB_ts[::,fd_width]
            x = 0.25*(BB0[0]+BB0[2]+BB2[0]+BB2[2])
            y = 0.25*(BB0[1]+BB0[3]+BB2[1]+BB2[3])
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
        time1 = time.clock()
        gen = make_generator(test_scene, x, v, dt, Nt)
        print "Time: {}".format(time1 - time.clock())
        print "starting eval"
        #iterate through all time steps
        ct = 0
        # while 1:
        #     print ct
        #     pr = cProfile.Profile()
        #     pr.enable()
        #     bleherg = next(gen, None)
        #     pr.disable()
        #     s = StringIO.StringIO()
        #     sortby = 'cumulative'
        #     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        #     ps.print_stats()
        #     if ct % 10 == 0:
        #         print s.getvalue()
        #         raw_input()
        #     ct += 1

        # assert False
        import time
        t = time.time()

        for (ct, data) in enumerate(gen):
            print time.time() - t
            #ignore predictions where actual data doesn't exist
            #Concatenate all xs, ps for the different classes
            with writer.saving(fig, "writer_test{}.mp4".format(i), 100):
                for thr in range(0, Nt-5, 5):
                    xs = np.array([[0,0]])
                    ps = np.array([])
                    #print "Evaluation for agent {}, time {}".format(i, dt * ct)
                    print thr
                    for cl in range(test_scene.num_nl_classes):
                        xys, weights = data[cl]
                        xys = xys[thr:thr+5]
                        weights = weights[thr:thr+5]
                        #show all weights
                        weights = weights.flatten()
                        p = weights#[where]
                        xy_xs = xys[:, 0, :].flatten()#[where]
                        xy_ys = xys[:, 1, :].flatten()#[where]
                        xy = np.array(zip(xy_xs, xy_ys))
                        if len(p) > 0:
                            ps = np.concatenate((ps, p))
                        if len(p) > 0:
                            xs = np.concatenate((xs, xy))
                        xys, weights = data[cl]
                        xys = xys[-1*(thr+5):-1*thr]
                        weights = weights[-1*(thr+5):-1*thr]
                        #show all weights
                        weights = weights.flatten()
                        p = weights#[where]
                        xy_xs = xys[:, 0, :].flatten()#[where]
                        xy_ys = xys[:, 1, :].flatten()#[where]
                        xy = np.array(zip(xy_xs, xy_ys))
                        if len(p) > 0:
                            ps = np.concatenate((ps, p))
                        if len(p) > 0:
                            xs = np.concatenate((xs, xy))

                    #delete placeholder component
                    xs = np.delete(xs, 0, 0)
                    #Define initial conditions
                    where = np.where(ps > 0)[0]
                    xs = xs[where]
                    ps = ps[where]
                    rho = (xs, ps)
                    tau = 0.01
                    lin_term = data[-1]
                    if len(xs) > 0:

                        stpx = 0.01
                        stpy = stpx
                        x, y = np.mgrid[slice(-test_scene.height/2, test_scene.height/2 + stpy, stpy),
                                        slice(-test_scene.width/2, test_scene.width/2 + stpx, stpx)]
                        pts = np.array([x.flatten(), y.flatten()])
                        vals = lin_term[thr](pts).reshape(x.shape)
                        #np.concatenate((xs, np.array([[0, 10]])))
                        mx = np.amax(ps)
                        mx_lin = np.amax(vals)
                        #mesh = plt.pcolormesh(x, y, vals, cmap="viridis", vmin = 0, vmax = mx_lin, alpha = 0.5)
                        mesh = plt.imshow(vals, cmap='viridis', vmin=0, vmax=mx_lin, zorder=10,
                                   extent=[-test_scene.width/2, test_scene.width/2, -test_scene.height/2, test_scene.height/2],
                                          interpolation='nearest', alpha = 0.35, origin="lower")
                        #c = np.asarray([(0, 0, 0, 0.5* v/mx) for v in ps])
                        #np.concatenate((c, np.array([[0, 0, 0, 1]])))
                        colors = cmap.to_rgba(ps/mx)
                        colors[:, 3] = ps/mx / 2
                        plt.scatter(xs[:, 0], xs[:, 1], c = colors, s = 3, cmap="viridis", edgecolors='none', zorder=11)
                        #plt.scatter(xs[:, 0], xs[:, 1], c=ps/mx, s = 3, cmap="viridis", edgecolors='none')
                        xs = [[],[]]
                        #DRAW MEASUREMENTS
                        #for i in range(100):
                        x1, v1 = get_initial_condition(test_BB_ts[:, (10 + thr):])
                        plt.scatter(x1[0], x1[1], c="white", s=1, edgecolors="none", zorder=12)
                        w = test_scene.bbox_width/2
                        lines = []
                        ln1, = plt.plot([x1[0] - w, x1[0] + w], [x1[1] - w, x1[1] - w], color='white', linestyle='-', linewidth=1, zorder=12)
                        ln2, = plt.plot([x1[0] - w, x1[0] + w], [x1[1] + w, x1[1] + w], color='white', linestyle='-', linewidth=1, zorder=12)
                        ln3, = plt.plot([x1[0] - w, x1[0] - w], [x1[1] - w, x1[1] + w], color='white', linestyle='-', linewidth=1, zorder=12)
                        ln4, = plt.plot([x1[0] + w, x1[0] + w], [x1[1] - w, x1[1] + w], color='white', linestyle='-', linewidth=1, zorder=12)
                        plt.ylim([-test_scene.height/2, test_scene.height/2])
                        plt.xlim([-test_scene.width/2, test_scene.width/2])
                        #plt.axes().set_aspect('equal', 'datalim')
                        plt.axis('off')
                        plt.imshow(img, zorder=0, extent=[-test_scene.width/2, test_scene.width/2, -test_scene.height/2, test_scene.height/2])
                        writer.grab_frame()
                        ln1.remove()
                        ln2.remove()
                        ln3.remove()
                        ln4.remove()
                        mesh.remove()
                        #plt.clf()
                plt.clf()

            # def lin(x, y):
            #     x = x.flatten()
            #     y = y.flatten()
            #     return lin_term([x,y])
            # print "Starting evaluate_plane"
            # resolution = test_scene.bbox_width
            # #Define rho_true for a given time step etc
            # rt = lambda bboxes: rho_true(i, ct/10+12, test_set,bboxes)
            # bbox = np.array([test_scene.width, test_scene.height])
            # #Call evaluate_plane
            # res =  evaluate_plane(bbox, rho, rt, tau, lin, resolution, debug=True)
            # xs = [[], []]
            # plt.scatter(xs[0], xs[1], s=10, color="green")
            # plt.savefig('foo.png')

            # print res


