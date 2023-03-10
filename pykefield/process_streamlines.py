import numpy as np
import pandas as pd

from .utils import get_radius, get_theta


def make_streamline(func_3d, start, dL=0.1,
                    active_boundaries=(0, 664, -1480.3, -11),
                    boundary_type='rz',
                    v=False, output=None):
    '''Function to compute the streamline of the Electric field
    from a specific point. Finishes when the "electron" gets
    out of the active volume, default to cylindrical nT TPC.

    Parameters:
      * func_3d - list of 3D interpolative functions to consider
      (Ex_3d,Ey_3d,Ez_3d,Emod_3d)
      * start - (x,y,z) point where to start the computation
      * active_boundaries - where to stop computation. Defaults
      to the nT active volume cylinder.
      * boundary_type: 'rz', (r_min, r_max, z_min, z_max), or
      'xyz', (x_min, x_max, y_min, y_max, z_min, z_max).
      * v - verbose level
      * output - str with type of output desired:
          - 'streamline': return pd.DataFrame with each entry a
          computed point along the path
          - 'change': returns a np.array containing the correction
          in r,theta,z from the inital position to the corrected one.
          The z is the length of the path plus the distance from
          the last z_slice computed in the 3d functions.
          - 'None': returns the corrected position of the given point.
    Returns:
      * Depends on the 'output' parameters (see above).'''

    # Get streamline:
    #     Define starting r
    #     Calculate n= E/|E|
    #     Calculate r' = r + ndL
    #     Save r' in array or df
    #     Check if r' is still inside TPC. if not -> finish
    #     Update r = r'

    # TO DO:
    #     Check if it's stalling, log and coninue. Rigth now
    # it just never finishes...

    _counter = 0
    L = 0
    Ex_3d, Ey_3d, Ez_3d, Emod_3d = func_3d
    start_r = get_radius(start[0], start[1])
    if output == 'streamline':
        df_streamline = pd.DataFrame({'x': [start[0]],
                                      'y': [start[1]],
                                      'z': [start[2]],
                                      'Ex': [Ex_3d(start)],
                                      'Ey': [Ey_3d(start)],
                                      'Ez': [Ez_3d(start)],
                                      'Emod': [Emod_3d(start)],
                                      'r': start_r
                                      }
                                     )

    _r = np.array(start)
    out_of_bounds = False
    while out_of_bounds == False:

        _r_next = get_next_r_streamline(_r, func_3d, dL, direction=-1)
        L += get_lenght_drifted(_r, _r_next)

        assert (~np.isnan(_r_next).any()),\
            'Lost the electron (computed fields is nan) at %s. Goodbye.' % (_r)

        _r_next_radius = get_radius(_r_next[0], _r_next[1])

        if output == 'streamline':
            df_streamline = df_streamline.append(
                {'x': _r_next[0],
                 'y': _r_next[1],
                 'z': _r_next[2],
                 'Ex': np.float64(Ex_3d(_r_next)),
                 'Ey': np.float64(Ey_3d(_r_next)),
                 'Ez': np.float64(Ez_3d(_r_next)),
                 'Emod': np.float64(Emod_3d(_r_next)),
                 'r': _r_next_radius
                }, ignore_index=True)

        out_of_bounds_gate, out_of_bounds_wall = bool_out_of_bounds(
            _r_next, active_boundaries, boundary_type)

        out_of_bounds = out_of_bounds_gate or out_of_bounds_wall

        if (out_of_bounds != False) and (v != False):
            print('The electron bumped into mean stuff and quit. Goodbye.')
#         if (v !=False) and (_counter%v== 0):
#             print('r before:', _r)
#             print('r after:', _r_next)
        _counter += 1
        assert ~(np.abs(_r - _r_next) < 0.00001).all(
        ), 'The electron is stuck. Either collected in the anode (Yay!) or on a numerical loop (Nay!)'

        if (v != False) and (_counter % v == 0):
            print('Current position(x,y,r,z): %.2f, %.2f, %.2f , %.2f' %
                  (_r_next[0], _r_next[1], _r_next_radius, _r_next[2]))
        _r = _r_next

    if (out_of_bounds_wall == True) and (v != False):
        print('Electron lost on wall.')

    if output == 'streamline':
        return df_streamline

    elif output == 'change_xyz':
        #        if out_of_bounds_wall == True:
        #            return np.array([np.nan, np.nan,np.nan])
        #        else:
        x_change = _r[0] - start[0]
        y_change = _r[1] - start[1]
        z_change = L - np.abs(start[2]) - 8

        change = [x_change, y_change, z_change]
        return change

    elif output == 'change_rz':
        r_change = get_radius(_r[0], _r[1]) - start_r
        theta_change = get_theta(_r[0], _r[1]) - get_theta(start[0], start[1])
        # go around due to z value missing from -11 upward #_r[2] - start[2]
        z_change = -((L + 12) - np.abs(start[2]))
        return np.array([r_change, theta_change, z_change], dtype=np.float32)

    else:
        _x_end = _r[0]
        _y_end = _r[1]
        _z_end = _r[2]
        _L_end = -(L - 8)
        return _x_end, _y_end, _z_end, _L_end


def get_lenght_drifted(r_before, r_after):
    """Calculate the distance between two points in a straight line."""
    r_diff = r_after - r_before
    dist = np.linalg.norm(r_diff)
    return dist


def bool_out_of_bounds(r, active_boundaries, boundary_type):
    '''Is this pos out of bounds and where from?
       - Reached the gate: (True, ??)
       - Lost on wall: (False, True)
       - No: (False,False)'''

    x = r[0]
    y = r[1]
    z = r[2]
    radius = get_radius(x, y)

    if boundary_type == 'rz':
        ans_gate = z > active_boundaries[3]
        ans_wall = ((r < active_boundaries[0]) or
                    (r > active_boundaries[1]))

    elif boundary_type == 'xyz':
        x_min, x_max, y_min, y_max, z_min, z_max = active_boundaries
        ans_wall = ((x < x_min) or
                    (x > x_max) or
                    (y < y_min) or
                    (y > y_max) or
                    (z < z_min) or
                    (z > z_max))

        ans_gate = False

    return (ans_gate, ans_wall)


def get_next_r_streamline(_r, func_3d, dL, direction):
    """Apply Efield to get next point in streamline."""
    Ex_3d, Ey_3d, Ez_3d, Emod_3d = func_3d

    # get field at specific point
    _Ex, _Ey, _Ez, _Emod = Ex_3d(_r), Ey_3d(_r), Ez_3d(_r), Emod_3d(_r)
    _Emod = np.linalg.norm(np.array([_Ex, _Ey, _Ez]))
    # calculate **n**
    n_x = _Ex / _Emod
    n_y = _Ey / _Emod
    n_z = _Ez / _Emod
    n_vec = np.concatenate((n_x, n_y, n_z))
    _r_next = _r + n_vec * dL * direction
    return _r_next
