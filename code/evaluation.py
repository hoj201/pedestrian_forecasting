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

def classifier(E, rho, tau, lin_term):
    #takes E: np.array(n_boxes, 2, 2)
    #looks like [[[box_width, box_height], [box_x, box_y]]]
    #rho: (np.array(n_points, 2), np.array(n_points))
    #tau: float
    #returns: bool array
    xy, p = rho

    def filter(e):
        fn = lambda x: (np.absolute((x[0]-e[1][0])) <= e[0][0]/2.0) and (np.absolute(x[1] - e[1][1]) <= e[0][1]/2.0)
        return sum(p[np.where(map(fn, xy))])

    res = np.array(map(filter, E))
    integrals = []
    for box in E:
        bounds = [box[1][0] - box[0][0]/2, box[1][0] + box[0][0]/2,
                  box[1][1] - box[0][1]/2, box[1][1] + box[0][1]/2]
        integrals.append(trap_quad(lin_term, bounds))

    integrals = np.array(integrals)

    return res + integrals > tau

def true_classifier(E, rho_true, tau):
    #takes:
    #E: [[[box_width, box_height], [box_x, box_y]]]
    #rho_true: function
    #tau: float
    #returns: bool array
    integrals = []
    for box in e:
        bounds = [box[1][0] - box[0][0]/2, box[1][0] + box[0][0]/2,
                  box[1][1] - box[0][1]/2, box[1][1] + box[0][1]/2]
        integrals.append(trap_quad(rho_true, bounds))
    integrals = np.array(integrals)
    return integrals > tau


def precision(pred, truth):
    return len(np.where(np.logical_and(pred, truth))[0]) / float(len(np.where(truth)[0]))

def recall(pred, truth):
    return len(np.where(np.logical_and(pred, truth))[0]) / float(len(np.where(pred)[0]))

def accuracy(pred, truth):
    return len(np.where(pred == truth)[0]) / float(len(truth))

def evaluate_plane(bbox, rho, rho_true, tau, lin_term, resolution):
    #Takes:
    #bbox: np.array(2): [scene_width, scene_height]
    #rho: (np.array(n_points, 2), np.array(n_points))
    #rho_true: function
    #tau: float
    #lin_term: function
    #resolution: np.array(2): [num_boxes_x, num_boxes_y]

    #returns (precision, recall, accuracy)

    x_vals = np.linspace(-bbox[0]/2, bbox[0]/2, resolution[0]+1)
    y_vals = np.linspace(-bbox[1]/2, bbox[1]/2, resolution[1]+1)
    bboxes = []
    for x in resolution[0]:
        for y in resolution[1]:
            bboxes.append([[bbox[0]/resolution[0], bbox[1]/resolution[1]],
                  [x + bbox[0]/resolution[0]/2, y + bbox[1]/resolution[1]/2]])
    pred = classifier(bbox, rho, tau, lin_term)
    true = true_classifier(bbox, rho_true, tau)

    precision = precision(pred, true)
    recall = recall(pred, true)
    accuracy = accuracy(pred, true)

    return (precision, recall, accuracy)



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
    E = np.array([[[0.4, 0.4], [1,1]], [[1, 1], [0,0]], [[0.2, 0.2], [0.6, 0.6]]])
    def fn(x, y):
        x = x.flatten()
        return np.zeros(len(x))
    print classifier(E, rho, tau, fn)





