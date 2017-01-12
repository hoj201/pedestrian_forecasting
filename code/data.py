import pickle
with open("test_scene.pkl", "rb") as f:
    scene = pickle.load(f)

with open('test_set.pkl','r') as f:
    set = pickle.load(f)
