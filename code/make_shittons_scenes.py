from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
import pickle
import process_data
from scene import Scene
from sys import argv
import os
import json
from json_help import read_json
file = argv[1]
dic = read_json(file)

folder = dic['folder']
fname = dic['filename']
print "Initializing a scene from " + folder
BB_ts_list, width, height = process_data.get_BB_ts_list(folder,label="Biker")
params = read_json("params.json")
print params['nfold']
kf = KFold(n_splits = params['nfold'])

def mkdir(fname):
    try:
        os.mkdir(fname)
    except:
        pass
mkdir("scenes/" + fname)
mkdir("scenes/" + fname + "/scenes")
mkdir("scenes/" + fname + "/sets")
mkdir("scenes/" + fname + "/train_sets")

print [x.shape for x in BB_ts_list]

for j, (train_set, test_set) in enumerate(kf.split(BB_ts_list)):
    train_set = [x[1] for x in filter(lambda x:  x[0] in train_set, enumerate(BB_ts_list))]
    test_set = [x[1] for x in filter(lambda x:  x[0] in test_set, enumerate(BB_ts_list))]

    test_scene = Scene( train_set, width, height)
    #print "Display clusters"
    #for k in range( test_scene.num_nl_classes ):
    #    from visualization_routines import visualize_cluster
    #    visualize_cluster( test_scene, k )
    
    print "P(k) = "
    print test_scene.P_of_c
    
    print "\sum_k P(k) = {}".format( test_scene.P_of_c.sum() )
    print "Pickling scene and test set."
    with open("scenes/" + fname + "/scenes/{}.pkl".format(j), "w") as f:
        pickle.dump( test_scene, f)
        print "Pickled scene"
        
    with open("scenes/" + fname + "/sets/{}.pkl".format(j), "w") as f:
        pickle.dump( test_set, f)
        print "Pickled the test set with {} agents".format(len(test_set))
    with open("scenes/" + fname + "/train_sets/{}.pkl".format(j), "w") as f:
        pickle.dump( train_set, f)
        print "Pickled the test set with {} agents".format(len(train_set))
    
    
