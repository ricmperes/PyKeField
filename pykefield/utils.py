import numpy as np


def get_theta(x, y):
    theta = np.arctan(y / x)
    if (x < 0) and (y < 0):
        theta = -theta
    return theta


def get_radius(x, y):
    _r = np.sqrt(np.power(x, 2) + np.power(y, 2))
    return _r


def get_xy(r, theta):
    _x = r * np.cos(theta)
    _y = r * np.sin(theta)
    return _x, _y


def printmain(EFpoints_df):
    print(
        '''
        Computed field points: %d
             |   min   |   max
        --------------------
          x  | %.4f | %.4f
          y  | %.4f | %.4f
          z  | %.4f | %.4f
         Phi | %.2f | %.2f
         |E| | tbd | tbd
        ''' % (len(EFpoints_df),
               np.min(EFpoints_df.x), np.max(EFpoints_df.x),
               np.min(EFpoints_df.y), np.max(EFpoints_df.y),
               np.min(EFpoints_df.z), np.max(EFpoints_df.z),
               np.min(EFpoints_df.Phi), np.max(EFpoints_df.Phi)
               )
    )
    return None
