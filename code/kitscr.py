import numpy as np
import matplotlib.pyplot as plt
def raster(st, end):
    pixel_arr = []
    st = np.array(st)
    end = np.array(end) 
    offset = st
    rotation = [1 if st[0] < end[0] else -1, 1 if st[1] < end[1] else -1]
    tmpend = (end-st) * rotation
    ptst = np.array([0.5, 0.5])
    ptend = tmpend + np.array([0.5, 0.5])
    xs = np.array(range(1, tmpend[0]))
    alphas = xs/ptend[0]
    xs = np.array([0.5] + list(xs) + [end[0] + 0.5])
    ys = np.array([0.5] + list(alphas * ptend[1]) + [ptend[1]])
    for ind, y in list(enumerate(ys))[:len(ys)-1]:
        for j in range(int(y+1E-6), int(ys[ind+1]+1E-6)+1):
            pixel = [xs[ind], j]
            pixel_arr.append(pixel)
    pixel_arr.append(list(tmpend))
    for ind, pixel in enumerate(pixel_arr):
        pixel_arr[ind] = map(int, list((np.array(pixel) * rotation) + st))
    return pixel_arr

if __name__ == "__main__":
    st = [0, 0]
    end = [100, 0]
    pixels = raster(st, end)
    pixels = np.array(pixels).transpose()
    plt.plot([st[0], end[0]], [st[1], end[1]])

    plt.plot(pixels[0], pixels[1])
    plt.scatter(pixels[0], pixels[1])
    plt.show()
