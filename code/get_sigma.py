from data import sets
from sys import argv
import numpy as np
from process_data import BB_ts_to_curve as bbts
from train_random_walk import learn_sigma_RW
for ct, i in enumerate(sets):
    trgarf = map(bbts, i)
    print "sigma for {} is {}".format(ct, learn_sigma_RW(trgarf))

