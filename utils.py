import numpy as np

def get_theta(x,y):
    theta = np.arctan(y/x)
    if (x < 0) and (y < 0):
        theta = -theta
    return theta

def get_radius(x,y):
    _r = np.sqrt(np.power(x,2) + np.power(y,2))
    return _r

def get_xy(r, theta):
    _x = r * np.cos(theta)
    _y = r * np.sin(theta)
    return _x,_y