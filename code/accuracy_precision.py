import numpy as np

def accuracy( scene, BB_list , T_final):

    score = 0
    area = scene.V_scale[0] * scene.V_scale[1]
    for BB_time_series in BB_list:
        mu, eta = get_mu_eta( BB_time_series )
        scene.set_mu( mu )
        scene.set_eta( eta )
        BB = BB_time_series[T_final]
        # generate a grid on the bounding box.
        rho = scene.predict( X_grid, Y_grid, T_final )
        if rho.mean() / area > 0.05:
            score += 1
    return score
