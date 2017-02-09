from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
import pickle
import process_data
from scene import Scene
from sys import argv
if len(argv) > 1:
    folder = argv[1]
else:
    folder = '../annotations/coupa/video2/'
if len(argv) > 2:
    fname = argv[2]
else:
    fname = "test"
print "Initializing a scene from " + folder
BB_ts_list, width, height = process_data.get_BB_ts_list(folder,label="Biker")
kf = KFold(n_splits = 10)
train_set, test_set = train_test_split( BB_ts_list, random_state = 0 )
test_scene = Scene( train_set, width, height )
#print "Display clusters"
#for k in range( test_scene.num_nl_classes ):
#    from visualization_routines import visualize_cluster
#    visualize_cluster( test_scene, k )

print "P(k) = "
print test_scene.P_of_c

print "\sum_k P(k) = {}".format( test_scene.P_of_c.sum())
print "Pickling scene and test set."
with open("{}_scene.pkl".format(fname), "w") as f:
    pickle.dump( test_scene, f)
    print "Pickled scene"

with open("{}_set.pkl".format(fname), "w") as f:
    pickle.dump( test_set, f)
    print "Pickled the test set with {} agents".format(len(test_set))
