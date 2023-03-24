import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from tqdm import tqdm
from .common import *
from .plot_common import (get_axis_3d_paralelepipedo, get_axis_parallel_wires,
                          get_axis_perpendicular_wires)


def plot_perp_wires_streamlines(stream_list, df_field, all_geo_vars,
                                func=None, project='3d', save_fig=False, 
                                view_azi_angle=None,
                                figax = None):
    """Plot streamlines at the perpwires region."""
    x_min, x_max, y_min, y_max, z_min, z_max, x_gate, z_gate, x_anode1, x_anode2, z_anode = all_geo_vars
    if project == '3d':
        if figax is None:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = figax
        #ax = get_axis_3d_paralelepipedo(
        #    ax, x_min, x_max, y_min, y_max, z_min, z_max)
        ax = get_axis_perpendicular_wires(
            ax, x_gate, z_gate, x_anode1, x_anode2, z_anode, y_min, y_max)
        ax = get_axis_parallel_wires(
            ax, x_min, x_max, y_min, y_max, z_gate, z_anode)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_zlabel('z [mm]')

        ax.set_zlim(-50, 20)
        for _stream in stream_list:
            ax.plot(_stream.x, _stream.y, _stream.z, c='k',
                       marker='', lw = 0.8, alpha = 0.8
                       )
        ax.view_init(elev=20., azim=view_azi_angle)

        if save_fig != False:
            for _n, ii in enumerate(np.linspace(0, 360, 180)):
                ax.view_init(elev=20., azim=ii)
                fig.savefig(save_fig + "anim_%d.png" % (_n + 100))

    elif project == 'x0':
        if figax is None:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111)
        else:
            fig, ax = figax
        
        _x = 0.0
        _y = np.linspace(y_min, y_max, 500)
        _z = np.linspace(z_min, z_max, 500)
        _xxx, _yyy, _zzz = np.meshgrid(_x, _y, _z)

        cscale = ax.scatter(_yyy.flatten(), _zzz.flatten(), c=func((_xxx, _yyy, _zzz)),
                            cmap='Greys', alpha=0.6, marker='s', s=12, zorder=0)
        ax.set_xlabel('y [mm]')
        ax.set_ylabel('z [mm]')
        ax.set_xlim(y_min, y_max)
        ax.set_ylim(z_min, z_max)
        #fig.colorbar(cscale, label='Ez [kV/cm]')

        x_min = -50
        x_max = 30

        for _stream in stream_list:
            #cscale = ax.scatter(_stream.y, _stream.z, c=_stream.x, zorder=2,
            #                    vmin=x_min, vmax=x_max, marker='o', s=2)
            ax.plot(_stream.y, _stream.z, zorder = 2, 
                    ls = '-', marker = '', c = 'r', lw = 0.5)
        #fig.colorbar(cscale, label='x [mm]')

    elif project == 'y0':
        if figax is None:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111)
        else:
            fig, ax = figax

        _x = np.linspace(x_min, x_max, 500)
        _y = 0.0
        _z = np.linspace(z_min, z_max, 500)
        _xxx, _yyy, _zzz = np.meshgrid(_x, _y, _z)

        cscale = ax.scatter(_xxx.flatten(), _zzz.flatten(), 
                            c=func((_xxx, _yyy, _zzz)),
                            cmap='Greys', alpha=0.6, marker='s', 
                            s=12, zorder=0)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('z [mm]')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        #fig.colorbar(cscale, label='Ez [kV/cm]')

        y_min = -20
        y_max = 20

        for _stream in stream_list:
            cscale = ax.plot(_stream.x, _stream.z, 
                             color='r', zorder=2, lw=0.5)
        #plt.colorbar( cscale, label = 'y [mm]')

    if figax is not None:
        return fig,ax
    else:
        plt.show()


def plot_streamlines(stream_list, region=None):
    '''Plot a set of streamlines in RZ and XY
    Input:
      * stream_list - a list of all the pd.DataFrames with streamlines to plot.'''
    if region == 'perpwires':
        plot_perp_wires_streamlines(stream_list)
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))

    for _stream in stream_list:
        ax1.plot(_stream.r, _stream.z, c = 'k', alpha = 0.4, lw = 1)
    ax1.set_xlim(0, 670)
    ax1.set_ylim(-1501, 10) 
    TPC_r = Rectangle((0, z_min_TPC), r_TPC, z_max_TPC-z_min_TPC, 
                      color='k',fill=False, lw=2, 
                      label='TPC boundary')
    TPC_fiducial_r = Rectangle((0,fiducial_4t_zmin),
                        fiducial_4t_r,
                        fiducial_4t_zmax-fiducial_4t_zmin, 
                        color = 'red', 
                        lw = 2, fill = False,
                        zorder = 10,
                        label = '4t fiducial cylinder')
    ax1.add_patch(TPC_r)
    ax1.add_patch(TPC_fiducial_r)
    
    ax1.set_xlabel('r [mm]')
    ax1.set_ylabel('z [mm]')

    for _stream in stream_list:
        ax2.plot(_stream.x, _stream.y, c = 'k', alpha = 0.4, lw = 1)
    ax2.set_xlim(-670, 670)
    ax2.set_ylim(-670, 670)
    TPC_c = Circle((0, 0), r_TPC , color='k',
                   fill=False, lw=2, label='TPC boundary')
    
    TPC_fiducial_c = Circle((0, 0), fiducial_4t_r , color='red',
                   fill=False, lw=2, label='TPC boundary')
    
    ax2.add_patch(TPC_c)
    ax2.add_patch(TPC_fiducial_c)

    ax2.set_aspect('equal')
    
    ax2.set_xlabel('x [mm]')
    ax2.set_ylabel('y [mm]')

    if region == 'gate':
        ax1.set_ylim(-20, 2)

    plt.show()

def plot_2dstreamlines(Phi_3d, streamlist_xzplane, 
                       figax = None, save_fig = None):
    if figax is None:
        fig, ax = plt.subplots(1,1,figsize = (6,6))
    else:
        fig, ax = figax

    _y = 0.0
    _x = np.linspace(-650, 650, 100)
    _z = np.linspace(-1450, -20, 200)
    _xx, _yy, _zz = np.meshgrid(_x, _y, _z)

    cscale = ax.scatter(_xx.flatten(), _zz.flatten(), 
                        c = Phi_3d((_xx,_yy,_zz))/1000,
                        marker = 's', s = 12,
                        alpha = 0.6)
    
    rect_tpc = Rectangle((-fiducial_4t_r,fiducial_4t_zmin),
                            fiducial_4t_r*2,
                            fiducial_4t_zmax-fiducial_4t_zmin, 
                            color = 'yellow', 
                            lw = 2, fill = False,
                            zorder = 10,
                            label = '4t fiducial cylinder')
    ax.add_patch(rect_tpc)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('z [mm]')
    ax.set_xlim(-r_TPC,r_TPC)
    ax.set_ylim(-1460, -15)

    fig.colorbar(cscale, label = 'Potential [kV]')
    for _stream in streamlist_xzplane:
        ax.plot(_stream.x,_stream.z, ls = '-', 
                marker = '', c = 'k', lw = 1, 
                zorder = 10)
        
    if isinstance(save_fig, str):
        fig.savefig(save_fig)
    else:
        return fig, ax
    

def plot_3dstreamlines(streamlist, draw_tpc = True,
                       linecolor = 'k', 
                       figax = None, save_fig = None):
    if figax is None:
        fig, ax = plt.subplots(1,1,figsize = (6,6))
    else:
        fig, ax = figax

    for _stream in streamlist:
        ax.plot(_stream.x,_stream.y,_stream.z, lw = 1,
            ls = '-', marker = '', c = linecolor)


    if draw_tpc:
        # Cylinder
        theta_TPC = np.linspace(0, 2 * np.pi, 500)
        z_TPC_lin = np.linspace(-1500, 0, 1000)

        x_TPC = r_TPC * \
        np.outer(np.cos(theta_TPC), np.ones(np.size(z_TPC_lin)))
        y_TPC = r_TPC * \
        np.outer(np.sin(theta_TPC), np.ones(np.size(z_TPC_lin)))
        z_TPC = np.outer(np.ones(np.size(theta_TPC)), z_TPC_lin)

        # Draw parameters
        rstride = 20
        cstride = 10
        ax.plot_surface(x_TPC,y_TPC,z_TPC,alpha=0.1,rstride=rstride,
                        cstride=cstride,color='blue')
    
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_zlabel('z [mm]')
    ax.set_xlim(-700, 700)    
    ax.set_ylim(-700, 700)   
    ax.set_zlim(-1500, 0)

    if isinstance(save_fig, str):
        fig.savefig(save_fig)
    else:
        return fig, ax