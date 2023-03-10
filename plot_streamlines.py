

def plot_perp_wires_streamlines(stream_list, df_field, all_geo_vars,func = None, project = '3d', save_fig = False, view_azi_angle = None):
    x_min,x_max,y_min,y_max,z_min,z_max,x_gate, z_gate, x_anode1, x_anode2, z_anode = all_geo_vars
    if project == '3d':
        fig = plt.figure(figsize = (14,10))
        ax = fig.add_subplot(111, projection='3d')
        ax = get_axis_3d_paralelepipedo(ax,x_min,x_max,y_min,y_max,z_min,z_max)
        ax = get_axis_perpendicular_wires(ax,x_gate, z_gate, x_anode1, x_anode2, z_anode, y_min, y_max)
        ax = get_axis_parallel_wires(ax, x_min, x_max, y_min, y_max, z_gate, z_anode)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_zlabel('z [mm]')
        
        ax.set_zlim(-50,20)
        for _stream in stream_list:
            ax.scatter(_stream.x,_stream.y, _stream.z, c = 'k',
                       vmin = y_min , vmax = y_max, marker ='.', s = 1
                      )
        ax.view_init(elev=20., azim=view_azi_angle)
        
        if save_fig != False:
            for _n, ii in enumerate(np.linspace(0,360,180)):
                ax.view_init(elev=20., azim=ii)
                plt.savefig(save_fig + "anim_%d.png" %(_n+100))

    elif project == 'x0':
        fig = plt.figure(figsize = (16,10))
        _x = 0.0
        _y = np.linspace(y_min, y_max, 1000)
        _z = np.linspace(z_min, z_max, 800)
        _xxx,_yyy,_zzz = np.meshgrid(_x,_y,_z)
        
        ax = fig.add_subplot(111)
        cscale = ax.scatter(_yyy.flatten(),_zzz.flatten(), c = func((_xxx, _yyy, _zzz)),
                            cmap = 'Greys',alpha = 0.6,marker = 's', s = 6, zorder = 0)
        ax.set_title('Ez @ y=0')
        ax.set_xlabel('y [mm]')
        ax.set_ylabel('z [mm]')
        plt.colorbar(cscale, label = 'Ez [kV/cm]')
        
        x_min = -50
        x_max = 30
        
        for _stream in stream_list:
             cscale = ax.scatter(_stream.y,_stream.z, c = _stream.x, zorder = 2, 
                                 vmin = x_min, vmax = x_max, marker = 'o',s = 2)
        plt.colorbar( cscale, label = 'x [mm]')
        
    elif project == 'y0':
        fig = plt.figure(figsize = (16,10))
        _x = np.linspace(x_min, x_max, 1000)
        _y = 0.0
        _z = np.linspace(z_min, z_max, 800)
        _xxx,_yyy,_zzz = np.meshgrid(_x,_y,_z)
        
        ax = fig.add_subplot(111)
        cscale = ax.scatter(_xxx.flatten(),_zzz.flatten(), c = func((_xxx, _yyy, _zzz)),
                            cmap = 'Greys',alpha = 0.6,marker = 's', s = 6, zorder = 0)
        ax.set_title('Ez @ y=0')
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('z [mm]')
        plt.colorbar(cscale, label = 'Ez [kV/cm]')
        
        y_min = -20
        y_max = 20
        
        for _stream in stream_list:
             cscale = ax.plot(_stream.x,_stream.z, color = 'r',zorder = 2, lw = 0.5)
        #plt.colorbar( cscale, label = 'y [mm]')
        
    plt.show()

def plot_streamlines(stream_list, region = None):
    '''Plot a set of streamlines in RZ and XY
    Input:
      * stream_list - a list of all the pd.DataFrames with streamlines to plot.'''
    if region== 'perpwires':
        plot_perp_wires_streamlines(stream_list)
        return None
        
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,7))

    for _stream in stream_list:
        ax1.plot(_stream.r,_stream.z)
    ax1.set_xlim(0,670)
    ax1.set_ylim(-1501,10)
    TPC_r = matplotlib.patches.Rectangle((0,-1485),664,1485,color = 'red', fill = False,lw=2, label = 'TPC boundary')
    ax1.add_patch(TPC_r)

    for _stream in stream_list:
        ax2.plot(_stream.x,_stream.y)
    ax2.set_xlim(-664,664)
    ax2.set_ylim(-664,664)
    TPC_c = matplotlib.patches.Circle((0,0),0.664*1000,color = 'red', fill = False,lw=2, label = 'TPC boundary')
    ax2.add_patch(TPC_c)
    ax2.set_aspect('equal')
    
    if region == 'gate':
        ax1.set_ylim(-20,2)

    plt.show()

def get_anode_wires_y(y_min, y_max, y_reference_anode = -0.01304*1000):
    '''Returns the y coordinates of the anode wires from a 
    given y_min up to a given y_max. The reference point
    was taken directly form the mesh file by hand.
    The reference point comes twice, this is bug that I'm 
    too lazy right now to solve but shoudl be fine for ploting.
    Cheers!'''
    
    return np.concatenate(np.sort(np.arange(y_reference_anode, y_min, -5)), 
                          np.sort(np.arange(y_reference_anode, y_max, 5)))
    
def get_gate_wires_y(y_min, y_max, y_reference_gate = 0.007824*1000):
    '''Returns the y coordinates of the gate wires from a 
    given y_min up to a given y_max. The reference point
    was taken directly form the mesh file by hand.
    The reference point comes twice, this is bug that I'm 
    too lazy right now to solve but shoudl be fine for ploting.
    Cheers!'''

    return np.concatenate(np.sort(np.arange(y_reference_gate, y_min, -5)), 
                          np.sort(np.arange(y_reference_gate, y_max, 5)))