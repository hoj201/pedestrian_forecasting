from kitscr import raster
with open('../annotations/gates/video1/annotations.txt','r') as f:
    lines = f.readlines()

from sys import argv
width = int(argv[1])
new_width = int(argv[2])


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

def decode_line(line):
    sid, xL, yL, xR, yU, time, _, _, _, atype = line.split(' ')
    return int(sid), int(xL), int(yL), int(xR), int(yU), int(time), atype

def id_and_time(line):
    sid,_,_,_,_,time,_ = decode_line(line)
    return sid, time

virat_dict = {'"Pedestrian"\n':1, '"Biker"\n':4, '"Car"\n':2}
sid_current = None

runner_index = 0
duration = 0
lines = map(decode_line, lines)
sids = list(set([x[0] for x in lines]))
dic = dict(zip(sids, [[] for s in sids]))
[dic[x[0]].append(x) for x in lines]
import matplotlib.pyplot as plt
import numpy as np
names = open("virat/walk_basenames.txt", "w")

for sid in dic.keys():
    print sid
    t = 0
    #TIME, X, Y
    func = lambda x: (x[5], int((x[1] + x[3])/2), int((x[2] + x[4])/2))
    data_points = sorted( map(func, dic[sid]), key=lambda x: int(x[0]))
    px = []
    sts = []
    
    for inds, data in enumerate(data_points[1:]):
        ind = inds + 1
        start = [data_points[ind-1][1], data_points[ind-1][2]]
        sts.append(start)
        end = [data_points[ind][1], data_points[ind][2]]
        xs = raster(start, end)
        xs = xs[:len(xs)-1]
        for i in range(len(xs)):
            xs[i][0] = int(float(new_width)/width *xs[i][0])
            xs[i][1] = int(float(new_width)/width * xs[i][1])
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

    if len(pxs) > 15:
        with open("virat/virat_{}.txt".format(sid), "w") as f:
            f.write(st)
        names.write("virat_{}.txt".format(sid) + "\n")
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
