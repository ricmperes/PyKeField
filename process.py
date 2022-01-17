import numpy as np
import pandas as pd
from datetime import datetime
import scipy.interpolate as itp
from tqdm import tqdm
import matplotlib.pyplot as plt


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
        ''' %(len(EFpoints_df),
              np.min(EFpoints_df.x), np.max(EFpoints_df.x),
              np.min(EFpoints_df.y), np.max(EFpoints_df.y),
              np.min(EFpoints_df.z), np.max(EFpoints_df.z),
              np.min(EFpoints_df.Phi),np.max(EFpoints_df.Phi)
             )
    )
    return None

def get_finergrid_zslice(df_main, z_value, xx, yy, z_band = 0.01, plot_diff = False): 
    '''This function uses the method scipy.interpolate.Griddata
    to get the Phi values in a finer grid than in the output fo the
    simulation. Needs scipy.interpolate imported as itp.
    
    Parameters:
        df_main - a pd.DataFrame with at least x,y,z and Phi columns.
        z_value - z value of the slices to consider. 
        xx,yy - coordenates, in np.meshgrid form, of the points to compute
        z_band - the minimal distance between z slices in df_main for the 
    mask to select only one
        plot_diff = prints a plot showing the before and after. Not adaptative, 
    difference may not be noticeable
    
    Returns the Phi values on the final_grid coordenates.
    '''    
    #mask z_slice
    mask = (df_main.z < z_value+z_band) & (df_main.z > z_value-z_band)
    df = df_main[mask]
    #Get the initial Phi values
    vals = df.Phi
    #Some Phi values are np.nan. Lets mask these
    vals = np.ma.masked_invalid(vals)
    #Only the x and y coordinates with valid Phi are considered
    x_known = np.array(df.x)[~vals.mask]
    y_known = np.array(df.y)[~vals.mask]
    vals_known = vals[~vals.mask]
    interpPhi = itp.griddata((x_known, y_known), vals_known.ravel(),
                        (xx,yy), method = 'cubic')
    
    if plot_diff == True:
        plt.figure(figsize = (16,8))
        plt.subplot(121)
        plt.scatter(x = df.x, y = df.y, c = df.Phi, marker ='s',s = 1)
        plt.gca().set_aspect('equal')
        plt.colorbar()

        plt.subplot(122)
        plt.scatter(x = xx, y = yy, c = interpPhi, marker ='s',s = 1)
        plt.gca().set_aspect('equal')
        plt.colorbar()

        plt.show()        
        
    return interpPhi
    
def get_EField_zslice(df_main, z_list= None, 
                     xygridspecs = (-664,664,100,-664,664,100), 
                     plot_interp = False, plot_PhiE = False, plot_N = 5,
                    return_all = False):
    '''Function to process the Electric Potential values to get the 
    Electric Field values. Makes use of np.gradient method. Returns 
    both the 3D grid values for x,y,z, Phi values in that grid and 
    field values on the grid (Ex,Ey,Ez).
    
    Arguments:
        * df_main - a pd.DataFrame with at least x,y,z and Phi columns.
        * z_list - list of z values to consider.
        * xygridspecs - a tuple of the form (x_min, x_max, x_nb, y_min, 
    y_max, y_nb), defaults to (-664,664,100,-664,664,100)
        * plot_interp - boolean. Default False, if true plots initial 
    and final xy grid, colored with Phi values for each z slice.
        * plt_PhiE - boolean. Defaults to False, if true plots side by 
    side the Phi and |E| scatter plots for plot_N z slices (at least 
    first and last).
    
    Returns: 
        * pd.DataFrame with each row being a interpolated point.
    Columns: x,y,z,r2,Phi,Ex,Ey,Ez,E_mod
    '''
    x_min, x_max, x_nb, y_min, y_max, y_nb = xygridspecs
    
    _x = np.linspace(x_min,x_max,x_nb)
    _y = np.linspace(y_min,y_max,y_nb)
    if z_list != None:
        _z = z_list
    else:
        _z = np.unique(df_main.z)
    
    xx,yy = np.meshgrid(_x,_y, indexing = 'ij')
    xxx,yyy,zzz = np.meshgrid(_x,_y,_z, indexing = 'ij')
    
    Phi_array = np.zeros((len(_x),len(_y),len(_z)))

    for z_n, z_value in tqdm(enumerate(_z), 
                             'Computing Phi values in each z_slice',
                             total = len(_z)):
        corrected_Phi = get_finergrid_zslice(df_main = df_main, z_value = z_value,
                                            xx = xx, yy = yy, plot_diff= plot_interp)
        Phi_array[:,:,z_n] = corrected_Phi
    
    Ex, Ey, Ez = np.gradient(Phi_array, _x, _y, _z)
    Ex = -Ex
    Ey = -Ey
    Ez = -Ez
    Emod_array = np.sqrt(np.power(Ex,2) + np.power(Ey,2) + np.power(Ez,2))

    if plot_PhiE == True:
        if plot_N < 2:
            plot_N = 2
            
        plot_n = [0] + np.random.randint(1,len(_z),plot_N-2).tolist() + [len(_z)-1]
        for _n in plot_n:
            fig = plt.figure(figsize =(20,8))

            plt.subplot(122)
            plt.scatter(xxx[:,:,_n].ravel(),yyy[:,:,_n].ravel(), c = Emod_array[:,:,_n].ravel(), marker ='s',s = 3)
            plt.colorbar(label = 'V/m')
            plt.gca().set_aspect('equal')
            plt.title('|E|')

            plt.subplot(121)
            plt.scatter(xxx[:,:,_n].ravel(),yyy[:,:,_n].ravel(), c = Phi_array[:,:,_n].ravel(), marker ='s',s = 3)
            plt.colorbar(label = 'V')
            plt.gca().set_aspect('equal')
            plt.title('Phi')

            fig.suptitle('z = %.2f mm' %zzz[0,0,_n],fontsize = 24)

            plt.show()
    
    if return_all == True:
        return xxx,yyy,zzz, Phi_array ,Ex,Ey,Ez, Emod_array
    else:
        # Put everything in a pd.DataFrame
        df_field = pd.DataFrame({'x':xxx.ravel(), 'y': yyy.ravel(), 'z':zzz.ravel(),
                                 'Phi':Phi_array.ravel(), 'Ex':Ex.ravel(), 'Ey':Ey.ravel(),
                                 'Ez':Ez.ravel(), 'E_mod':Emod_array.ravel()})

        df_field['r2'] = np.power(df_field.x,2) + np.power(df_field.y,2)
        return df_field
        
def getphimean_zslice(df_main, z_list= None, 
                      r2zgridspecs = (0., 666.**2, 200),
                     return_all = False):
    '''Given a data frame witht he EF and Phi calculated in a set of 
    points (grid, preferably), returns the correspondent 2D mean values 
    in R2Z space in a dataset.
    
    Arguments:
      * df_main - the pd.DataFrame with r2,z,Ex,Ey,Ez,E_mod,Phi.
      * z_list - list of z value to consider. distance more than 0.1mm or 
      does not handle correctly.
      * r2gridspecs - a tuple with (r2_min,r2_max,r2_nb), z values are
      taken seperatly in z_list or all z slices in df_main considered.
    
    Returns:
      * pd.DataFrame with r2, z, Ex[mean],Ey[mean],Ez[mean],Emod[mean],Phi[mean]
    '''

    # To make the rz (scatter) plot one needs the mean value of the Phi/Emod in each (r,z) coordinate instead of
    # the individual values at each (x,y,z)
    # We can do this by defining a grid on r,z, selecting events in each square and getting the mean. 
    # This is, however, an iterative process, not sure how to make it array-like.

    # In this particular case, since the z variable is always sliced and not interpolated, only r will be 
    # re-descretized, the z slices wil be taken as the z values to compute on
    
    r2_min, r2_max, r2_nb = r2zgridspecs
    r2_step = (r2_max - r2_min)/r2_nb
    r2_vals_mean = np.linspace(r2_min,r2_max, r2_nb) + np.ones(r2_nb)*r2_step/2
    z_vals_mean = np.unique(df_main.z)
    
    #make the grid
    rr2,zz = np.meshgrid(r2_vals_mean,z_vals_mean)

    #initialize mean arrays
    Emod_mean = np.empty((r2_nb, len(z_vals_mean)))
    Emod_mean[:] = np.nan
    Phi_mean = np.empty((r2_nb, len(z_vals_mean)))
    Phi_mean[:] = np.nan
    Ex_mean = np.empty((r2_nb, len(z_vals_mean)))
    Ex_mean[:] = np.nan
    Ey_mean = np.empty((r2_nb, len(z_vals_mean)))
    Ey_mean[:] = np.nan
    Ez_mean = np.empty((r2_nb, len(z_vals_mean)))
    Ez_mean[:] = np.nan

    #Main loop. Could it be done in array mode? Maybe, but this works and is not too slow
    #UPDATE: It's definetly too slow, needs improvement!! (14.07.2020)
    for _z_idx, _z in tqdm(enumerate(z_vals_mean[:-1]), 
                           'Computing mean values of Field and Phi in 2D projection',
                           total=len(z_vals_mean[:-1])):
        for _r2_idx, _r2 in enumerate(r2_vals_mean):
            _mask = (df_main.r2 >= _r2 - r2_step/2) & (df_main.r2 < _r2 + r2_step/2) & (df_main.z > _z-0.01) & (df_main.z < _z+0.01)
            _df = df_main[_mask]

            Ex_mean[_r2_idx, _z_idx] = np.mean(_df.Ex)
            Ey_mean[_r2_idx, _z_idx] = np.mean(_df.Ey)
            Ez_mean[_r2_idx, _z_idx] = np.mean(_df.Ez)
            Emod_mean[_r2_idx, _z_idx] = np.mean(_df.E_mod)
            Phi_mean[_r2_idx, _z_idx] = np.mean(_df.Phi)
    if return_all == True:
        return rr2, zz, Ex_mean, Ey_mean, Ez_mean, Emod_mean, Phi_mean
    else:
        #Get everything in a pd.DataFrame because they're cool and ezpz
        df_meanfield = pd.DataFrame({'r2':rr2.ravel(), 'z':zz.ravel(), 'Ex':Ex_mean.ravel('F') ,
                                     'Ey':Ey_mean.ravel('F') , 'Ez':Ez_mean.ravel('F') , 'Phi':Phi_mean.ravel('F') ,
                                     'Emod':Emod_mean.ravel('F') }) # Why is it inverted and have to use 'F' ordering?? No idea...
        return df_meanfield

def make_streamline(func_3d,start, dL=0.1,
                    active_boundaries = (0,664,-1480.3,-11),
                    v = False,output = None):
    '''Function to compute the streamline of the Electric field 
    from a specific point. Finishes when the "electron" gets
    out of the active volume, default to cylindrical nT TPC.
    
    Parameters:
      * func_3d - list of 3D interpolative functions to consider
      (Ex_3d,Ey_3d,Ez_3d,Emod_3d)
      * start - (x,y,z) point where to start the computation
      * active_boundaries - where to stop computation. Defaults
      to the nT active volume cylinder.
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
    
    #Get streamline:
    #     Define starting r
    #     Calculate n= E/|E|
    #     Calculate r' = r + ndL
    #     Save r' in array or df
    #     Check if r' is still inside TPC. if not -> finish
    #     Update r = r'
    
    #TO DO:
    #     Check if it's stalling, log and coninue. Rigth now
    # it just never finishes...
    
    _counter = 0
    L = 0
    Ex_3d,Ey_3d,Ez_3d,Emod_3d = func_3d
    start_r = get_radius(start[0],start[1])
    if output == 'streamline':
        df_streamline = pd.DataFrame({'x':[start[0]],
                                      'y':[start[1]],
                                      'z':[start[2]],
                                      'Ex':[Ex_3d(start)],
                                      'Ey':[Ey_3d(start)],
                                      'Ez':[Ez_3d(start)],
                                      'Emod':[Emod_3d(start)],
                                      'r':start_r
                                     }
                                    )
    
    _r = np.array(start)
    out_of_bounds = False
    while out_of_bounds == False:
        
        _r_next = get_next_r_streamline(_r,func_3d, dL,direction = -1)
        L += get_lenght_drifted(_r, _r_next)
        
        assert (~np.isnan(_r_next).any()),\
        'Lost the electron (computed fields is nan) at %s. Goodbye.'%(_r)
        
        _r_next_radius = get_radius(_r_next[0],_r_next[1])

        if output == 'streamline':
            df_streamline = df_streamline.append({'x':_r_next[0],
                                                  'y':_r_next[1],
                                                  'z':_r_next[2],
                                                  'Ex':np.float64(Ex_3d(_r_next)),
                                                  'Ey':np.float64(Ey_3d(_r_next)),
                                                  'Ez':np.float64(Ez_3d(_r_next)),
                                                  'Emod':np.float64(Emod_3d(_r_next)),
                                                  'r':_r_next_radius
                                                 }, ignore_index=True)
        
        out_of_bounds_gate, out_of_bounds_wall = bool_out_of_bounds(_r_next_radius,_r_next[2], active_boundaries)
        out_of_bounds = out_of_bounds_gate or out_of_bounds_wall
        if (out_of_bounds != False) and (v != False):
            print('The electron bumped into mean stuff and quit. Goodbye.')
        
        _counter+=1
        
        if (v !=False) and (_counter%10== 0):
            print('Current position(x,y,r,z): %.2f, %.2f, %.2f , %.2f' %(_r_next[0],_r_next[1],_r_next_radius, _r_next[2]))
        _r = _r_next
        
    if (out_of_bounds_wall == True) and (v!=False):
        print('Electron lost on wall.')
        
    if output == 'streamline':
        return df_streamline
    
    if output == 'change':
        if out_of_bounds_wall == True:
            return np.array([np.nan, np.nan,np.nan])
        else:
            r_change = get_radius(_r[0],_r[1]) - start_r
            theta_change = get_theta(_r[0],_r[1]) - get_theta(start[0], start[1])
            z_change = -((L + 12) - np.abs(start[2])) # go around due to z value missing from -11 upward #_r[2] - start[2]
            return np.array([r_change, theta_change, z_change],dtype = np.float32)
    
    else:
        if out_of_bounds_wall == True:
            return np.array([np.nan, np.nan,np.nan])
        else:
            _x_end = _r[0]
            _y_end = _r[1]
            _L_end = -(L + 12)
            return np.array([_x_end,_y_end,_L_end],dtype = np.float32)


def get_lenght_drifted(r_before, r_after):
    r_diff = r_after-r_before
    dist = np.linalg.norm(r_diff)
    return dist

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

def bool_out_of_bounds(r,z, active_boundaries):
    '''Is this pos out of bounds and where from?
       - Reached the gate: (True, ??)
       - Lost on wall: (False, True)
       - No: (False,False)'''
    ans_gate = z > active_boundaries[3]
    ans_wall = ((r < active_boundaries[0]) or 
                (r > active_boundaries[1]))
    
    return (ans_gate, ans_wall)

def get_next_r_streamline(_r,func_3d,dL,direction):
    Ex_3d,Ey_3d,Ez_3d,Emod_3d = func_3d
    
    #get field at specific point
    _Ex,_Ey,_Ez,_Emod = Ex_3d(_r),Ey_3d(_r),Ez_3d(_r),Emod_3d(_r)
    _Emod = np.linalg.norm(np.array([_Ex,_Ey,_Ez]))
    #calculate **n**
    n_x = _Ex/_Emod
    n_y = _Ey/_Emod
    n_z = _Ez/_Emod
    n_vec = np.concatenate((n_x,n_y,n_z))
    _r_next = _r + n_vec * dL * direction
    return _r_next