import numpy as np
from derived_posteriors import joint_lin_x_t_x_hat_v_hat as fun


x = np.array([0, 0])
v = np.array([-0.01, 0.01])
t = 100

xs = np.linspace(-1, 1, 100)
XS, YS = np.meshgrid(xs, xs)
pts = np.vstack((XS.flatten(), YS.flatten()))

ret = fun(t, pts, x, v)
fig = plt.figure()
plt.scatter(pts[0], pts[1], c=ret)
plt.show()

