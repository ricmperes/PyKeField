import numpy as np
import pandas as pd
from datetime import datetime
import scipy.interpolate as itp
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib


def plot_averageEF(df_main,var = 'mod',r2_cut_minus = 0,r2_cut_plus = 664**2, z_cut_minus = -1480,z_cut_plus = -9):
    '''Plot the intensity of the Electric field in r2_z given a set range '''
    
    V_nT = np.pi * r2_cut_plus/100 * (z_cut_plus/10-z_cut_minus/10) #cm**3
    d_Xe = 2.862 #g/cc
    W_nT = d_Xe * V_nT *0.001 #kg

    _mask = (df_main.r2 >= r2_cut_minus) & (df_main.r2 < r2_cut_plus) & (df_main.z < z_cut_plus) & (df_main.z >= z_cut_minus)
    _df = df_main[_mask]
    #_df['r'] = np.sqrt(_df['r2'])
    
    plt.figure(figsize=(8,7))
    if var=='mod':
        c = _df.Emod
    elif var=='x':
        c = _df.Ex
    elif var == 'y':
        c = _df.Ey
    elif var == 'z':
        c = _df.Ez
    else:
        raise ValueError("The var you're looking for is in another castle.")
        
    plt.scatter(_df.r2,_df.z, c = c,marker = 's', s = 40)

    plt.colorbar(label = '|E| [V/mm]')
    plt.xlabel('R2 [mm^2]')
    plt.ylabel('Z [mm]')
    plt.title('Electric Field %s - fiducial volume: %.2f t'%(var,W_nT/1000))
    plt.text((r2_cut_plus+r2_cut_minus)/2,(z_cut_plus+z_cut_minus)/2, 
             'Average |E|: %.2f V/mm' %np.mean(c),horizontalalignment='center', fontsize=12)

    plt.show()

def plot_xy(x,y,series_to_plot, acc_grid = False, title = None,TPC_line = True):
    fig,ax = plt.subplots(1,figsize = (10,10))
    
    plt.scatter(x,y,c = series_to_plot)
    plt.title(title)
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    #plt.xlim(-710,710)
    #plt.ylim(-710,710)
    plt.colorbar(label = 'Potential')
    #TPC = matplotlib.patches.Rectangle((0,-1485),665**2,1485+10,color = 'red', fill = False,lw=2)
    if TPC_line == True:
        TPC = matplotlib.patches.Circle((0,0),664,color = 'red', fill = False,lw=2)
        ax.add_patch(TPC)
    # if acc_grid == True:
    #     low_acc_list, med_acc_list, high_acc_list = get_acc_lists()
    #     acc_grid_list = get_acc_grid_list(low_acc_list, med_acc_list, high_acc_list)
    #     for patch in acc_grid_list:
    #         ax.add_patch(patch)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    plt.show()