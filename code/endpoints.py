import numpy as np
import data
from process_data import BB_ts_to_curve
import matplotlib.pyplot as plt

scene = data.scenes[0]
for cl in scene.clusters:
    beginx = 0
    beginy = 0
    endx = 0
    endy = 0
    for agent in cl:
        beginx += agent[0, 0]
        beginy += agent[1, 0]
        endx += agent[0, -1]
        endy += agent[1, -1]
    begin = np.array([beginx, beginy]) / len(cl)
    end = np.array([endx, endy])/ len(cl)
    print begin
    print end

