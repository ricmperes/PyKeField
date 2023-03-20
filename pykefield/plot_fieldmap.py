import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import FuncFormatter
import pandas as pd

from .common import *


def plot_averageEF(df_main, var='mod', r2_cut_minus=0,
                   r2_cut_plus=664**2, z_cut_minus=-1480, z_cut_plus=-9):
    '''Plot the intensity of the Electric field in r2_z given a set range '''

    V_nT = np.pi * r2_cut_plus / 100 * \
        (z_cut_plus / 10 - z_cut_minus / 10)  # cm**3
    d_Xe = 2.862  # g/cc
    W_nT = d_Xe * V_nT * 0.001  # kg

    _mask = (
        df_main.r2 >= r2_cut_minus) & (
        df_main.r2 < r2_cut_plus) & (
            df_main.z < z_cut_plus) & (
                df_main.z >= z_cut_minus)
    _df = df_main[_mask]
    #_df['r'] = np.sqrt(_df['r2'])

    plt.figure(figsize=(8, 7))
    if var == 'mod':
        c = _df.Emod
    elif var == 'x':
        c = _df.Ex
    elif var == 'y':
        c = _df.Ey
    elif var == 'z':
        c = _df.Ez
    else:
        raise ValueError("The var you're looking for is in another castle.")

    plt.scatter(_df.r2, _df.z, c=c, marker='s', s=40)

    plt.colorbar(label='|E| [V/mm]')
    plt.xlabel('R2 [mm^2]')
    plt.ylabel('Z [mm]')
    plt.title(
        'Electric Field %s - fiducial volume: %.2f t' %
        (var, W_nT / 1000))
    plt.text((r2_cut_plus + r2_cut_minus) / 2, (z_cut_plus + z_cut_minus) / 2,
             'Average |E|: %.2f V/mm' % np.mean(c), horizontalalignment='center', fontsize=12)

    plt.show()


def plot_xy(x, y, series_to_plot, acc_grid=False, title=None, TPC_line=True):
    fig, ax = plt.subplots(1, figsize=(10, 10))

    plt.scatter(x, y, c=series_to_plot)
    plt.title(title)
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    # plt.xlim(-710,710)
    # plt.ylim(-710,710)
    plt.colorbar(label='Potential')
    #TPC = Rectangle((0,-1485),665**2,1485+10,color = 'red', fill = False,lw=2)
    if TPC_line == True:
        TPC = Circle(
            (0, 0), 664, color='red', fill=False, lw=2)
        ax.add_patch(TPC)
    # if acc_grid == True:
    #     low_acc_list, med_acc_list, high_acc_list = get_acc_lists()
    #     acc_grid_list = get_acc_grid_list(low_acc_list, med_acc_list, high_acc_list)
    #     for patch in acc_grid_list:
    #         ax.add_patch(patch)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    plt.show()

    ## slice in x
# make grid


def plot_x_slice_perpwires(_x, func, y_min, y_max,
                           z_min, z_max, save_fig=False):
    """Plot yz plane with perpwires."""
    fig = plt.figure(figsize=(12, 14))
    ax = fig.add_subplot(111, projection='3d')

    _y = np.linspace(y_min, y_max, 100)
    _z = np.linspace(z_min, z_max, 200)
    _xxx, _yyy, _zzz = np.meshgrid(_x, _y, _z)

    sclicescatter = ax.scatter(
        _xxx, _yyy, _zzz, c=func(
            (_xxx, _yyy, _zzz)), marker='s', s=20)
    ax.set_xlim(-700, 700)
    ax.set_ylim(-700, 700)
    ax.set_zlim(-1510, 20)

    ax.set_title('x=%.2f; %s' % (_x, func.name))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if func.name == 'Phi':
        cbar_label = '%s [V]' % func.name
    else:
        cbar_label = '%s [V/mm]' % func.name
    cbar = fig.colorbar(sclicescatter, shrink=0.5, label=cbar_label)
    if save_fig != False:
        plt.savefig('./figures/animation_xslice_%d' % save_fig)
    else:
        plt.show()


def plot_y_slice(_y, func, save_fig=False):
    """Plot a xz plane."""
    fig = plt.figure(figsize=(12, 14))
    ax = fig.add_subplot(111, projection='3d')
    _x = np.linspace(-664, 664, 100)
    _z = np.linspace(-1500, -20, 200)
    _xxx, _yyy, _zzz = np.meshgrid(_x, _y, _z)
    sclicescatter = ax.scatter(
        _xxx, _yyy, _zzz, c=func(
            (_xxx, _yyy, _zzz)), marker='s', s=20)
    ax.set_xlim(-700, 700)
    ax.set_ylim(-700, 700)
    ax.set_zlim(-1510, 20)
    ax.set_title('x=%.2f; %s' % (_y, func.name))
    # _TPC_rect = Rectangle((-r_TPC*1000,-1500),
    #                                         r_TPC*1000*2,1510,
    #                                         color = 'k', fill = False,lw=2, alpha =0.8, label = 'TPC')
    # ax.add_patch(_TPC_rect)
    # plt.colorbar()
    # Cylinder
    theta_TPC = np.linspace(0, 2 * np.pi, 500)
    z_TPC_lin = np.linspace(-1500, 0, 1000)
    x_TPC = r_TPC * 1000 * \
        np.outer(np.cos(theta_TPC), np.ones(np.size(z_TPC_lin)))
    y_TPC = r_TPC * 1000 * \
        np.outer(np.sin(theta_TPC), np.ones(np.size(z_TPC_lin)))
    z_TPC = np.outer(np.ones(np.size(theta_TPC)), z_TPC_lin)
    # Draw parameters
    rstride = 20
    cstride = 10
    ax.plot_surface(
        x_TPC,
        y_TPC,
        z_TPC,
        alpha=0.2,
        rstride=rstride,
        cstride=cstride,
        color='blue')
    #ax.plot_surface(Xc_TPC, -Yc_TPC, Zc_TPC, alpha=0.2, rstride=rstride, cstride=cstride,  color = 'blue')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if func.name == 'Phi':
        cbar_label = '%s [V]' % func.name
    else:
        cbar_label = '%s [V/mm]' % func.name
    cbar = fig.colorbar(sclicescatter, shrink=0.5, label=cbar_label)
    if save_fig != False:
        plt.savefig('./figures/animation_yslice_%d' % save_fig)
    else:
        plt.show()


def plot_z_slice(_z, func, save_fig=False):
    """Plot a Z plane."""
    fig = plt.figure(figsize=(12, 14))
    ax = fig.add_subplot(111, projection='3d')

    # Cylinder
    theta_TPC = np.linspace(0, 2 * np.pi, 500)
    z_TPC_lin = np.linspace(-1500, 0, 1000)
    x_TPC = r_TPC * 1000 * \
        np.outer(np.cos(theta_TPC), np.ones(np.size(z_TPC_lin)))
    y_TPC = r_TPC * 1000 * \
        np.outer(np.sin(theta_TPC), np.ones(np.size(z_TPC_lin)))
    z_TPC = np.outer(np.ones(np.size(theta_TPC)), z_TPC_lin)
    # Draw parameters
    rstride = 20
    cstride = 10
    ax.plot_surface(
        x_TPC,
        y_TPC,
        z_TPC,
        alpha=0.1,
        rstride=rstride,
        cstride=cstride,
        color='blue')
    #ax.plot_surface(Xc_TPC, -Yc_TPC, Zc_TPC, alpha=0.2, rstride=rstride, cstride=cstride,  color = 'blue')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    _x = np.linspace(-664, 664, 100)
    _y = np.linspace(-664, 664, 100)
    _xxx, _yyy, _zzz = np.meshgrid(_x, _y, _z)
    sclicescatter = ax.scatter(
        _xxx, _yyy, _zzz, c=func(
            (_xxx, _yyy, _zzz)), marker='s', s=20)
    ax.set_xlim(-700, 700)
    ax.set_ylim(-700, 700)
    ax.set_zlim(-1510, 20)
    ax.set_title('x=%.2f; %s' % (_z, func.name))
    # _TPC_rect = Rectangle((-r_TPC*1000,-1500),
    #                                         r_TPC*1000*2,1510,
    #                                         color = 'k', fill = False,lw=2, alpha =0.8, label = 'TPC')
    # ax.add_patch(_TPC_rect)
    # plt.colorbar()

    if func.name == 'Phi':
        cbar_label = '%s [V]' % func.name
    else:
        cbar_label = '%s [V/mm]' % func.name
    cbar = fig.colorbar(sclicescatter, shrink=0.5, label=cbar_label)

    if save_fig != False:
        plt.savefig('./figures/animation_zslice_%d' % save_fig)
    else:
        plt.show()


def plot_xy_slice(_x_set, _y_set, func, save_fig=False):
    """Plot xz and yz planes at a given crossing a."""
    fig = plt.figure(figsize=(12, 14))
    ax = fig.add_subplot(111, projection='3d')

    _y = np.linspace(_y_set, 664, 50)
    _z = np.linspace(-1500, -20, 200)
    _xxx, _yyy, _zzz = np.meshgrid(_x_set, _y, _z)
    c_x = func((_xxx, _yyy, _zzz))
    sclicescatter_x = ax.scatter(_xxx, _yyy, _zzz, 
                                 c=c_x, marker='s',
                                 vmin=np.nanmin(c_x), 
                                 vmax=np.nanmax(c_x),
                                 s=20, zorder=1)

    _x = np.linspace(-664, 664, 100)
    _z = np.linspace(-1500, -20, 200)
    _xxx, _yyy, _zzz = np.meshgrid(_x, _y_set, _z)
    c_y = func((_xxx, _yyy, _zzz))
    sclicescatter_y = ax.scatter(_xxx, _yyy, _zzz, c=c_y,
                                 marker='s', s=20,
                                 vmin=np.nanmin(c_x), 
                                 vmax=np.nanmax(c_x),
                                 zorder=2)

    _y = np.linspace(-664, _y_set, 50)
    _z = np.linspace(-1500, -20, 200)
    _xxx, _yyy, _zzz = np.meshgrid(_x_set, _y, _z)
    c_x = func((_xxx, _yyy, _zzz))
    sclicescatter_x2 = ax.scatter(_xxx, _yyy, _zzz, 
                                  c=c_x, marker='s', 
                                  s=20, 
                                  vmin=np.nanmin(c_x), 
                                  vmax=np.nanmax(c_x), 
                                  zorder=3)

    ax.set_xlim(-700, 700)
    ax.set_ylim(-700, 700)
    ax.set_zlim(-1510, 20)

    #ax.set_title('x=%.2f; %s' %(_x, func.name))
    # _TPC_rect = Rectangle((-r_TPC*1000,-1500),
    #                                         r_TPC*1000*2,1510,
    #                                         color = 'k', fill = False,lw=2, alpha =0.8, label = 'TPC')
    # ax.add_patch(_TPC_rect)
    # plt.colorbar()

    # Cylinder
    theta_TPC = np.linspace(0, 2 * np.pi, 500)
    z_TPC_lin = np.linspace(-1500, 0, 1000)

    x_TPC = r_TPC * 1000 * \
        np.outer(np.cos(theta_TPC), np.ones(np.size(z_TPC_lin)))
    y_TPC = r_TPC * 1000 * \
        np.outer(np.sin(theta_TPC), np.ones(np.size(z_TPC_lin)))
    z_TPC = np.outer(np.ones(np.size(theta_TPC)), z_TPC_lin)

    # Draw parameters
    rstride = 20
    cstride = 10
    ax.plot_surface(
        x_TPC,
        y_TPC,
        z_TPC,
        alpha=0.2,
        rstride=rstride,
        cstride=cstride,
        color='blue')
    #ax.plot_surface(Xc_TPC, -Yc_TPC, Zc_TPC, alpha=0.2, rstride=rstride, cstride=cstride,  color = 'blue')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if func.name == 'Phi':
        cbar_label = '%s [V]' % func.name
    else:
        cbar_label = '%s [V/mm]' % func.name
    cbar = fig.colorbar(sclicescatter_x, shrink=0.5, label=cbar_label)
    if save_fig != False:
        plt.savefig('./figures/xy_slice_%d' % save_fig)
    else:
        plt.show()



def plot_average_phi(rr2: np.ndarray, zz: np.ndarray, 
                     Phi_mean: np.ndarray, figax = None,
                     save_fig = False):
    if figax is None:
        fig, ax = plt.subplots(1,1,figsize = (6,6))
    else:
        fig, ax = figax

    c = Phi_mean.ravel('F')

    sct = ax.scatter(rr2.ravel(),
                     zz.ravel(),
                     c= c,
                     marker = 's', s = 12,
                     )

    lines = np.arange(-30000,0,1000)
    ax.contour(rr2, zz, Phi_mean.T, lines, 
            alpha = 0.5,  linewidths  = 1,
            colors = 'k')
    
    rect_tpc = Rectangle((0,fiducial_4t_zmin),
                        fiducial_4t_r**2,
                        fiducial_4t_zmax-fiducial_4t_zmin, 
                        color = 'yellow', 
                        lw = 2, fill = False,
                        zorder = 10,
                        label = '4t fiducial cylinder')


    ax.set_xlim(0,(r_TPC)**2)
    #xticks_linear = np.array([0,200,300,400,500,600])
    #ax.set_xticks(xticks_linear**2, xticks_linear)
    ax.set_xlabel('r$^2$ [mm]$^2$')

    ax.set_ylim(-1490, -15)
    ax.set_ylabel('z [mm]')

    ax.add_patch(rect_tpc)
    #ax.legend(loc = 'upper left')
    cbar = fig.colorbar(sct, label = 'E. potential [kV]')
    cbar.set_ticks(np.linspace(np.nanmin(c), np.nanmax(c), 7))
    cbar.set_ticklabels(np.linspace(-30, 0, 7))
    
    if isinstance(save_fig, str):
        fig.savefig(save_fig)
    else:
        return fig, ax


def plot_average_efield(rr2: np.ndarray, 
                        zz: np.ndarray, 
                        EE: np.ndarray,
                        df_meanfield: pd.DataFrame, 
                        efieldtype:str = 'Ez',
                        fidcolor: bool = False,
                        figax: tuple = None,
                        colorlabel = '',
                        save_fig: str = False):
    if figax is None:
        fig, ax = plt.subplots(1,1,figsize = (6,6))
    else:
        fig, ax = figax

    if fidcolor == True:
        _fiducial_df_meanfield = df_meanfield[
            (df_meanfield['r2'] < (fiducial_4t_r)**2) &
            (df_meanfield['z'] > fiducial_4t_zmin) & 
            (df_meanfield['z'] < fiducial_4t_zmax)
            ]

        sct = ax.scatter(df_meanfield['r2'],
                 df_meanfield['z'],
                 c = df_meanfield[efieldtype]*10,
                 vmin = min(_fiducial_df_meanfield[efieldtype])*10, 
                 vmax = max(_fiducial_df_meanfield[efieldtype])*10, 
                 marker = 's', s = 12, cmap = 'viridis')
    else:
        sct = ax.scatter(df_meanfield['r2'],
                         df_meanfield['z'],
                         c = df_meanfield[efieldtype]*10,
                         marker = 's', s = 12)
    
    manual_locations = [(100000, -280), (100000, -500), (100000, -1000), 
                        (100000, -1150), (100000, -1300), (200000, -180),
                        (300000, -500)]
    
    if efieldtype == 'Emod':
        lines = np.linspace(175,190,4)
        contour = ax.contour(rr2, zz, EE.T*10, lines,
                                alpha = 0.8,  linewidths  = 1,
                                colors = 'k')
        ax.clabel(contour, contour.levels, inline=True, 
                    fmt = fmt_contour, fontsize = 8,
                    manual = manual_locations, inline_spacing = 20)

    elif efieldtype == 'Ez':
        lines = np.linspace(-190,-175,4)
    
        contour = ax.contour(rr2, zz, EE.T*10, lines,
                                alpha = 0.8,  linewidths  = 1,
                                colors = 'k')
        ax.clabel(contour, contour.levels, inline=True, 
                    fmt = fmt_contour, fontsize = 8,
                    manual = manual_locations, inline_spacing = 20)

    rect_tpc = Rectangle((0,fiducial_4t_zmin),
                        fiducial_4t_r**2,
                        fiducial_4t_zmax-fiducial_4t_zmin, 
                        color = 'yellow', 
                        lw = 2, fill = False,
                        zorder = 10,
                        label = '4t fiducial cylinder')


    ax.set_xlim(0,(r_TPC)**2)

    ax.set_xlabel('r$^2$ [mm]$^2$')

    ax.set_ylim(-1490, -15)
    ax.set_ylabel('z [mm]')

    ax.add_patch(rect_tpc)

    cbar = fig.colorbar(sct, label = colorlabel)
    # cbar.set_ticks(np.linspace(np.nanmin(c), np.nanmax(c), 7))
    # cbar.set_ticklabels(
    #     [fmt_colorlabel(boop) for boop in np.linspace(np.nanmin(c)*10, 
    #                                                   np.nanmax(c)*10, 
    #                                                   7)
    #     ])

    
    if isinstance(save_fig, str):
        fig.savefig(save_fig)
    else:
        return fig, ax

def fmt_colorlabel(x):
    return f'{x:.1f}'

def fmt_contour(x):
    return f'{x*10:.1f} V/cm'
