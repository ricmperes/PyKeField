import numpy as np
import pandas as pd
from datetime import datetime
import scipy.interpolate as itp
from tqdm import tqdm
import matplotlib.pyplot as plt

from PyKeField.utils import *

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


def get_interp_functions(df_regulargrid):
    '''Returns the continuous accessible 3d functions for Ex, Ey, Ez, 
    Emod and Phi. These functions are interpolated from the given points
    of df (need a regular grid)'''

    N_x = len(np.unique(df_regulargrid.x))
    N_y = len(np.unique(df_regulargrid.y))
    N_z = len(np.unique(df_regulargrid.z))
    xxx = np.array(df_regulargrid.x).reshape(N_x,N_y,N_z)
    yyy = np.array(df_regulargrid.y).reshape(N_x,N_y,N_z)
    zzz = np.array(df_regulargrid.z).reshape(N_x,N_y,N_z)
    Ex = np.array(df_regulargrid.Ex).reshape(N_x,N_y,N_z)
    Ey = np.array(df_regulargrid.Ey).reshape(N_x,N_y,N_z)
    Ez = np.array(df_regulargrid.Ez).reshape(N_x,N_y,N_z)
    Phi_array = np.array(df_regulargrid.Phi).reshape(N_x,N_y,N_z)
    Emod_array = np.array(np.sqrt(np.power(df_regulargrid.Ex,2) + 
                                np.power(df_regulargrid.Ey,2) + 
                                np.power(df_regulargrid.Ez,2))).reshape(N_x,N_y,N_z)
    
    Phi_3d = itp.RegularGridInterpolator((xxx[:,0,0], yyy[0,:,0], zzz[0,0,:]), Phi_array,bounds_error=False, fill_value=np.nan)
    Emod_3d = itp.RegularGridInterpolator((xxx[:,0,0], yyy[0,:,0], zzz[0,0,:]), Emod_array,bounds_error=False, fill_value=np.nan)
    Ex_3d = itp.RegularGridInterpolator((xxx[:,0,0], yyy[0,:,0], zzz[0,0,:]), Ex,bounds_error=False, fill_value=np.nan)
    Ey_3d = itp.RegularGridInterpolator((xxx[:,0,0], yyy[0,:,0], zzz[0,0,:]), Ey,bounds_error=False, fill_value=np.nan)
    Ez_3d = itp.RegularGridInterpolator((xxx[:,0,0], yyy[0,:,0], zzz[0,0,:]), Ez,bounds_error=False, fill_value=np.nan)
    Phi_3d.name = 'Phi'
    Emod_3d.name = 'Emod'
    Ex_3d.name = 'Ex'
    Ey_3d.name = 'Ey'
    Ez_3d.name = 'Ez'
    
    return Ex_3d,Ey_3d,Ez_3d,Emod_3d, Phi_3d