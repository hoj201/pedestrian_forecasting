import numpy as np
from data import scene
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

sigma_x = scene.sigma_x

def particle_generator(x_hat, v_hat, t_final, n_steps):
    """Generates a test distribution according to
    advecting a gaussian forward by all velocities
    on the line segment between v_hat and -v_hat
    It's really fucking stupid, but I'm wondering if it will look
    pretty similar in a lot of cases."""
    res = 20

    x_span = np.linspace(-sigma_x * 3, sigma_x * 3, res) + x_hat[0]
    y_span = np.linspace(-sigma_x * 3, sigma_x * 3, res) + x_hat[1]
    x,y = np.meshgrid(x_span, y_span)

    pts_org = np.vstack((x.flatten(), y.flatten()))
    vals_org = multivariate_normal.pdf(np.transpose(pts_org), mean=x_hat, cov=sigma_x**2)

    vals = np.zeros(res**2 * n_steps)
    vals[0 : res**2] = np.array(vals_org)
    pts = np.array(pts_org)
    ret = np.zeros((2, res**2 * n_steps))

    vals_back = np.zeros(res**2 * n_steps)
    vals_back[0 : res**2] = np.array(vals_org)
    pts_back = np.array(pts_org)
    ret_back = np.zeros((2, res**2 * n_steps))

    for i in range(n_steps):
        t = float(t_final/n_steps) * i
        pts_t = np.array(pts_org)
        pts_t[0] += v_hat[0] * t
        pts_t[1] += v_hat[1] * t
        ret[:, i * res**2 : (i+1) * res**2] = pts_t
        vals[i * res**2 : (i+1) * res**2] = vals_org

        pts_t_1 = np.array(pts_org)
        pts_t_1[0] -= v_hat[0] * t
        pts_t_1[1] -= v_hat[1] * t
        ret_back[:, i * res**2 : (i+1) * res**2] = pts_t_1
        vals_back[i * res**2 : (i+1) * res**2] = vals_org

        yield np.hstack((ret, ret_back)), np.hstack((vals, vals_back))
