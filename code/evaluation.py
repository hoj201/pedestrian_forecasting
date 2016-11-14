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

def classifier(E, rho, tau, lin_term):
    E = np.array(E)
    #takes E: np.array(n_boxes, 2, 2)
    #looks like [[[box_width, box_height], [box_x, box_y]]]
    #rho: (np.array(n_points, 2), np.array(n_points))
    #tau: float
    #returns: bool array #NOTE hoj:  This does not output a bool, as written
    xy, p = rho
    asdf = {"counter": 0}
    def filter(e): #NOTE hoj:  Probably should rename this, as filter is a built-in func
        fn = lambda x: (np.absolute((x[0]-e[1][0])) <= e[0][0]/2.0 + 10E-6) and (np.absolute(x[1] - e[1][1]) <= e[0][1]/2.0 + 10E-6)
        asdf["counter"] += 1
        print "{}%".format(100*asdf["counter"] / float(len(E)))
        return sum(p[np.where(map(fn, xy))[0]])
    res = np.array(map(filter, E))
    integrals = []
    for box in E:
        bounds = [box[1][0] - box[0][0]/2, box[1][0] + box[0][0]/2,
                  box[1][1] - box[0][1]/2, box[1][1] + box[0][1]/2]
        integrals.append(trap_quad(lin_term, bounds, res=(40, 40)))
    integrals = np.array(integrals)
    #print (res + integrals)[np.where(res + integrals > tau)[0]][36:]
    #print E[np.where(res + integrals > tau)[0]]
    return res + integrals

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
    return len(np.where(np.logical_and(pred, truth))[0]) / float(len(np.where(truth)[0]))

def recall(pred, truth):
    return len(np.where(np.logical_and(pred, truth))[0]) / float(len(np.where(pred)[0]))

def accuracy(pred, truth):
    return len(np.where(pred == truth)[0]) / float(len(truth))

def evaluate_plane(bbox, rho, rho_true, tau, lin_term, width):
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
    print "beginning prediction classifier"
    #create all of the bounding boxes
    ctx = int(np.ceil(bbox[0]/width))
    cty = int(np.ceil(bbox[1]/width))
    for x in range(ctx):
        for y in range(cty):
            bboxes.append([[width, width],
                           [-1 * bbox[0]/2.0 + width/2.0 + width * x, -1 * bbox[1]/2.0 + width/2.0 + width * y]])
    bboxes = np.array(bboxes)
    #Create prediction
    predic = classifier(bboxes, rho, tau, lin_term)
    pred = predic > tau
    print "beginning true classifier"
    #create truth for comparison
    truth = true_classifier(bboxes, rho_true, tau)
    true = truth > tau
    #pres = precision(pred, true)
    #rec = recall(pred, true)
    #acc = accuracy(pred, true)
    predic = predic[np.where(pred)[0]]
    truth = truth[np.where(true)[0]]

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

    bboxes_x = bboxes[:, :, 0].flatten()[1::2]
    bboxes_y = bboxes[:, :, 1].flatten()[1::2]
    #plt.scatter(bboxes_x, bboxes_y, color="yellow")
    weights_x = rho[0][:, 0]
    weights_y = rho[0][:, 1]
    plt.scatter(weights_x, weights_y, color="black", s = 5)

    #plt.show()

    #return (pres, rec, acc, pred, true)



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
