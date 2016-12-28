import scene
from scene import Scene
import numpy as np
import pickle
import process_data

if __name__ == "__main__":
    print "Testing initializer"
    
    folder = '../annotations/coupa/video2/'
    BB_ts_list, width, height = process_data.get_BB_ts_list(folder,label="Biker")

    #curve_ls = map( process_data.BB_ts_to_curve, BB_ts_list )
    from sklearn.cross_validation import train_test_split
    train_set, test_set = train_test_split( BB_ts_list, random_state = 0 )

    test_scene = Scene( train_set, width, height )
    print "Display clusters"
    for k in range( test_scene.num_nl_classes ):
        from visualization_routines import visualize_cluster
        visualize_cluster( test_scene, k )
   
    print "P(k) = "
    print test_scene.P_of_c

    print "\sum_k P(k) = {}".format( test_scene.P_of_c.sum())
    response = raw_input("Would you like to pickle this scene? y/n")
    if response == "y":
        with open("test_scene.pkl", "ws") as f:
            pickle.dump( test_scene, f)

        with open("test_set.pkl", "ws") as f:
            pickle.dump( test_set, f)
