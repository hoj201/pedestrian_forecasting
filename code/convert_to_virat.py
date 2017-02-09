from kitscr import raster
import os
import matplotlib.pyplot as plt
import numpy as np
from data import all_data
from process_data import BB_ts_to_curve as bbts

import json

with open('scene_order.json') as f:
    st = f.read()
json_acceptable_string = st.replace("'", "\"")
dic = json.loads(json_acceptable_string)
order = dic['order']

"""
    The format of the Stanford Drone Dataset is:
    #TODO: Maybe this is wrong
    subject_id x_L, x_R, y_L, y_U, time-stamp blah blah blah type

    The VIRAT dataset has two annotation files: objects and events

    The format of the VIRAT dataset object annotations is:
    subject_id duration time-stamp x_L y_U width height type

    for the types bike=4, person=1, car=2

    The format of the VIRAT dataset event annotation is:
    ... I think this is irrelevent... hopefully
"""


def mkdir(fname):
    try:
        os.mkdir(fname)
    except:
        pass


for sn, scene in enumerate(all_data):
    scene_name = order[sn]
    print scene_name
    mkdir("virat/" + scene_name)
    width = 200
    for trial, (test_scene, _, train_set) in enumerate(scene):
        mkdir("virat/" + scene_name + "/{}".format(trial))

        runner_index = 0
        duration = 0

        names = open("virat/{}/{}/walk_basenames.txt".format(scene_name, trial), "w")
        for ind, agent in enumerate(train_set):
            agent = bbts(agent)
            agent[0] += test_scene.width/2.0
            agent[1] = (-1 * agent[1] + test_scene.height/2)
            agent *= 200.0
            data_points = [(int(x), int(y)) for x,y in list(agent.transpose())]
            t = 0
            #TIME, X, Y



            px = []
            sts = []


            for inds, data in enumerate(data_points[1:]):
                ind = inds + 1
                start = [data_points[ind-1][0], data_points[ind-1][1]]
                sts.append(start)
                end = [data_points[ind][0], data_points[ind][1]]
                xs = raster(start, end)
                xs = xs[:len(xs)-1]
                for i in range(len(xs)):
                    xs[i][0] = int(xs[i][0])
                    xs[i][1] = int(xs[i][1])
                px += xs

            st = ""
            runs = []
            last = xs[0]
            in_run = False
            run_start = 0
            for ind, x in enumerate(px):
                if last == x and not in_run:
                    in_run = True
                    run_start = ind
                if last != x and in_run:
                    runs.append([run_start, ind])
                    in_run = False
                last = x
            pxs = []
            start = 0
            for run in runs:
                if start < run[0]:
                    pxs += px[start:run[0]]
                if run[1] - run[0] % 2 == 0:
                    elems = px[run[1]-1:run[0]]
                else:
                    elems = px[run[1]:run[0]]
                for i in range(0, (len(elems)-1)/2):
                    elems[2*i + 1][0] +=1
                pxs += elems
                start = run[1]

            for x in pxs:

                st += "{} {} {} {} {}".format(t, x[0],  x[1], 0, 0) + "\n"
                t += 1

            #pts = np.array(pxs).transpose()
            #plt.plot(pts[0], pts[1])
            #plt.show()

            #inc = raw_input("should include? ")
            #if len(inc) == 0:
            #    print "not including"
            #    continue
            with open("virat/{}/{}/{}.txt".format(scene_name, trial, ind), "w") as f:
                f.write(st)
            names.write("{}.txt".format(ind) + "\n")
        names.close()




        #for line in lines:
        #    sid, xL, yL, xR, yU, time, atype = line
        #    print "XL: {} XR: {} YD: {} YU{}".format(xL, xR, yL, yU)
        #    x = int((xL + xR)/2)
        #    y = int((yL + yR)/2)#
        #
        #    if sid != sid_current:
        #        sid_current = sid
        #        sid, t_initial = id_and_time( lines[runner_index])
        #        while sid is sid_current:
        #            runner_index +=1
        #            sid,_ = id_and_time(lines[runner_index])
        #        runner_index -= 1
        #        _,t_final = id_and_time( lines[runner_index] )
        #        duration = t_final - t_initial
        #    width = xR-xL
        #    height = yU-yL
        #    vtype = virat_dict[atype]
        #    elements = map(str, [sid, duration, time, xL, yU, width, height, vtype])
        #    string = " ".join(elements)
        #    f_virat.write(string + '\n')

        #f_virat.close()
