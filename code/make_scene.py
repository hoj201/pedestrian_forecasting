from sklearn.cross_validation import train_test_split
import pickle
import process_data
from scene import Scene
folder = '../annotations/coupa/video2/'
print "Initializing a scene from " + folder
BB_ts_list, width, height = process_data.get_BB_ts_list(folder,label="Biker")
train_set, test_set = train_test_split( BB_ts_list, random_state = 0 )
test_scene = Scene( train_set, width, height )
print "Display clusters"
for k in range( test_scene.num_nl_classes ):
    from visualization_routines import visualize_cluster
    visualize_cluster( test_scene, k )

print "P(k) = "
print test_scene.P_of_c

print "\sum_k P(k) = {}".format( test_scene.P_of_c.sum())
print "Pickling scene and test set."
with open("test_scene.pkl", "w") as f:
    pickle.dump( test_scene, f)
    print "Pickled scene"

with open("test_set.pkl", "w") as f:
    pickle.dump( test_set, f)
    print "Pickled the test set"
