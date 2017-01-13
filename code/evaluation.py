#Accuracy = Pr(truth=pred)
#should be computed as:
     #sum(truth == pred)/len(E)
#Precision = Pr(pred|truth)
#Computed as:
     #sum(truth and pred)/ sum(truth)
#recall = pr(truth|pred)
#computed as:
     #sum(truth and pred) / sum(pred)
import numpy as np
from integrate import trap_quad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def classifier(bounds, width, rho, tau):
    #takes:
    #bounds: [width, height]
    #rho: (np.array(n_points, 2), np.array(n_points))
    #tau: float
    #returns: bool array #NOTE hoj:  This does not output a bool, as written
    pts, weights = rho
    asdf = {"counter": 0}

    indices = np.argsort(pts[0])
    pts = pts[:,indices]
    weights = weights[indices]

    ctx = int(np.ceil(bounds[0]/width))
    cty = int(np.ceil(bounds[1]/width))

    x_lims = np.linspace(-bounds[0]/2, bounds[0]/2, ctx+1)
    y_lims = np.linspace(-bounds[1]/2, bounds[1]/2, cty+1)

    bboxes = []
    sums = []

    for i in range(ctx):
        lbx = x_lims[i]
        ubx = x_lims[i+1]
        start = pts[0].searchsorted(lbx)
        end = pts[0].searchsorted(ubx)
        pts_x = pts[:,start:end]
        weights_x = weights[start:end]

        #Sort with respect to y-component
        indices = np.argsort(pts_x[1])
        pts_x = pts_x[:,indices]
        weights_x = weights_x[indices]
        for j in range(cty):
            lby = y_lims[j]
            uby = y_lims[j+1]
            start = pts_x[1].searchsorted(lby)
            end = pts_x[1].searchsorted(uby)
            sums.append(weights_x[start:end].sum())
            bboxes.append([[width, width],
                           [-1 * bounds[0]/2.0 + width/2.0 + width * i, -1 * bounds[1]/2.0 + width/2.0 + width * j]])

    #print(res + integrals)[np.where(res + integrals > tau)[0]][36:]
    #printE[np.where(res + integrals > tau)[0]]
    sums = np.array(sums)
    bboxes = np.array(bboxes)
    return sums, bboxes

def true_classifier(E, rho_true, tau):
    #takes:
    #E: [[[box_width, box_height], [box_x, box_y]]]
    #rho_true: function
    #tau: float
    #returns: bool array
    #integrals = []
    #for box in E:
    #    bounds = [box[1][0] - box[0][0]/2, box[1][0] + box[0][0]/2,
    #              box[1][1] - box[0][1]/2, box[1][1] + box[0][1]/2]
    #    integrals.append(trap_quad(rho_true, bounds, res=(40, 40)))
    #integrals = np.array(integrals)
    E = np.array(E)
    integrals = rho_true(E)
    #print E[np.where(integrals > tau)[0]]
    return integrals


def precision(pred, truth):
    if float(len(np.where(truth)[0])) != 0:
        return len(np.where(np.logical_and(pred, truth))[0]) / float(len(np.where(truth)[0]))
    else:
        return 1

def recall(pred, truth):
    if float(len(np.where(pred)[0])) != 0:
        return len(np.where(np.logical_and(pred, truth))[0]) / float(len(np.where(pred)[0]))
    else:
        return 1

def accuracy(pred, truth):
    return len(np.where(pred == truth)[0]) / float(len(truth))

def evaluate_plane(bbox, rho, rho_true, tau, width, debug_level=0):
    #Takes:
    #bbox: np.array(2): [scene_width, scene_height]
    #rho: (np.array(n_points, 2), np.array(n_points))
    #rho_true: function
    #tau: float
    #lin_term: function
    #width: float
    #returns (precision, recall, accuracy)
    xy,p = rho
    bboxes = []
    #create all of the bounding boxes
    #Create prediction
    predic, bboxes = classifier(bbox, width, rho, tau)
    pred = predic > tau
    #create truth for comparison
    truth = true_classifier(bboxes, rho_true, tau)
    true = truth > tau
    pres = precision(pred, true)
    rec = recall(pred, true)
    acc = accuracy(pred, true)
    predic = predic[np.where(pred)[0]]
    truth = truth[np.where(true)[0]]

    if debug_level > 0:
        #plotting bollocks
        true_bboxes = bboxes[np.where(true)[0]]
        pred_bboxes = bboxes[np.where(pred)[0]]#[34:]
        true_xs = true_bboxes[:, :, 0].flatten()[1::2]
        true_ys = true_bboxes[:, :, 1].flatten()[1::2]
        pred_xs = pred_bboxes[:, :, 0].flatten()[1::2]
        pred_ys = pred_bboxes[:, :, 1].flatten()[1::2]
        for box in pred_bboxes:
            x0 = box[1][0] - box[0][0]/2.0
            x1 = box[1][0] + box[0][0]/2.0
            y = box[1][1] - box[0][1]/2.0
            plt.plot([x0, x1], [y, y], color='purple', linestyle='-', linewidth=.5)
            x = box[1][0] + box[0][0]/2.0
            y0 = box[1][1] - box[0][1]/2.0
            y1 = box[1][1] + box[0][1]/2.0
            plt.plot([x, x], [y0, y1], color='purple', linestyle='-', linewidth=.5)
        for box in true_bboxes:
            x0 = box[1][0] - box[0][0]/2.0
            x1 = box[1][0] + box[0][0]/2.0
            y = box[1][1] - box[0][1]/2.0
            plt.plot([x0, x1], [y, y], color='orange', linestyle='-', linewidth=.5)
            x = box[1][0] + box[0][0]/2.0
            y0 = box[1][1] - box[0][1]/2.0
            y1 = box[1][1] + box[0][1]/2.0
            plt.plot([x, x], [y0, y1], color='orange', linestyle='-', linewidth=.5)
        plt.scatter(true_xs, true_ys, color="red", s=10)
        plt.scatter(pred_xs, pred_ys, color="blue", s=10)
        if debug_level > 1:
            plt.show()

    return (pres, rec, acc)# pred, true)



if __name__ == "__main__":
    import itertools
    rand1 = np.random.random([1000]) > 0.5
    print "Precision test 1:"
    print precision(rand1, rand1)
    print "Recall test 1:"
    print recall(rand1, rand1)
    print "Accuracy test 1:"
    print accuracy(rand1, rand1)

    #test 4 bounding box exhaustively
    #it = itertools.product("01", repeat = 8)
    #for i in it:
    #    fn = lambda x: x == "1"
    #    b1 = map(fn, i[0:4])
    #    b2 = map(fn, i[4:])
    #    try:
    #        print precision(b1, b2)
    #    except:
    #        pass
    #    try:
    #        print recall(b1, b2)
    #    except:
    #        pass
    #    try:
    #        print accuracy(b1, b2)
    #    except:
    #        pass


    #Test dirac deltas

    tau = 0.001
    rho = (np.array([[0.5, 0.5], [-0.5, -0.5]]), np.array([0.2, 0.1]))
    E = np.array([[[.01, .01], [.5, .495]], [[.01, .01], [-.5, -.495]]])
    def fn(x, y):
        x = x.flatten()
        return np.zeros(len(x))
    print "should return [False, True, True]:"
    print classifier(E, rho, tau, fn)

    assert False

    bbox = np.array([0.4, 0.4])
    tau = 0.1
    rho = (np.array([[-0.1, 0.1], [0.11, 0.11], [0, 0], [-.1, -.1], [-.1, 0]]), np.array([0.2, 0.2, 0.2, .2, .2]))

    lin_term = lambda x, y: np.zeros(len(x.flatten()))

    #def rho_true(x, y):
    #    x = x.flatten()
    #    y = y.flatten()
    #    filt = lambda y0: 1 if (np.absolute(y0[0] - 0.1) <= 0.02) and (np.absolute(y0[1] - 0.1) <= 0.02) else 0
    #    ident = np.array(map(filt, zip(x, y)))
    #    return 1.0/0.04**2 * ident
    def rho_true(bounding_boxes):
        return np.zeros([len(bounding_boxes)])

    resolution = [2, 2]

    print evaluate_plane(bbox, rho, rho_true, tau, lin_term, resolution)
