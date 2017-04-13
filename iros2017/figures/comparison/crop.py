from sys import argv
import numpy as np
import cv2
name = argv[1]
height1 = int(argv[2])
height2 = int(argv[3])
img = cv2.imread(name)
print img.shape
img2 = img[height1:height2, :, :]
indp = name.index(".")
fname = name[:indp] + "_crop" + name[indp:]
cv2.imwrite(fname, img2)

