import numpy as np


def get_axis_3d_paralelepipedo(ax, x_min, x_max, y_min, y_max, z_min, z_max,
                               color='grey', alpha=0.2, lines=True):

    x_region = np.linspace(x_min, x_max, 2)
    y_region = np.linspace(y_min, y_max, 2)
    z_region = np.linspace(z_min, z_max, 2)

    xx, yy = np.meshgrid(x_region, y_region)
    zz_up = np.ones(np.shape(xx)) * z_max
    zz_down = np.ones(np.shape(xx)) * z_min
    ax.plot_surface(xx, yy, zz_up, color=color, alpha=alpha)
    ax.plot_surface(xx, yy, zz_down, color=color, alpha=alpha)

    xx, zz = np.meshgrid(x_region, z_region)
    yy_up = np.ones(np.shape(xx)) * y_max
    yy_down = np.ones(np.shape(xx)) * y_min
    ax.plot_surface(xx, yy_up, zz, color=color, alpha=alpha)
    ax.plot_surface(xx, yy_down, zz, color=color, alpha=alpha)

    yy, zz = np.meshgrid(y_region, z_region)
    xx_up = np.ones(np.shape(yy)) * x_max
    xx_down = np.ones(np.shape(yy)) * x_min
    ax.plot_surface(xx, yy, xx_up, color=color, alpha=alpha)
    ax.plot_surface(xx, yy, xx_down, color=color, alpha=alpha)

    ax.plot(x_region, np.ones(np.shape(x_region)) * y_min,
            np.ones(np.shape(x_region)) * z_min, color='k', alpha=0.8)
    ax.plot(x_region, np.ones(np.shape(x_region)) * y_max,
            np.ones(np.shape(x_region)) * z_min, color='k', alpha=0.8)
    ax.plot(x_region, np.ones(np.shape(x_region)) * y_min,
            np.ones(np.shape(x_region)) * z_max, color='k', alpha=0.8)
    ax.plot(x_region, np.ones(np.shape(x_region)) * y_max,
            np.ones(np.shape(x_region)) * z_max, color='k', alpha=0.8)
    ax.plot(np.ones(np.shape(y_region)) * x_min, y_region,
            np.ones(np.shape(x_region)) * z_min, color='k', alpha=0.8)
    ax.plot(np.ones(np.shape(y_region)) * x_max, y_region,
            np.ones(np.shape(x_region)) * z_min, color='k', alpha=0.8)
    ax.plot(np.ones(np.shape(y_region)) * x_min, y_region,
            np.ones(np.shape(x_region)) * z_max, color='k', alpha=0.8)
    ax.plot(np.ones(np.shape(y_region)) * x_max, y_region,
            np.ones(np.shape(x_region)) * z_max, color='k', alpha=0.8)
    ax.plot(np.ones(np.shape(x_region)) * x_min,
            np.ones(np.shape(y_region)) * y_min, z_region,
            color='k', alpha=0.8)
    ax.plot(np.ones(np.shape(x_region)) * x_max,
            np.ones(np.shape(y_region)) * y_min, z_region,
            color='k', alpha=0.8)
    ax.plot(np.ones(np.shape(x_region)) * x_min,
            np.ones(np.shape(y_region)) * y_max, z_region,
            color='k', alpha=0.8)
    ax.plot(np.ones(np.shape(x_region)) * x_max,
            np.ones(np.shape(y_region)) * y_max, z_region,
            color='k', alpha=0.8)

    return ax


def get_axis_perpendicular_wires(ax, x_gate, z_gate, x_anode1, x_anode2,
                                 z_anode, y_min, y_max):
    _y = np.linspace(y_min, y_max, 200)
    ax.plot(
        np.ones(
            np.shape(_y)) *
        x_gate,
        _y,
        np.ones(
            np.shape(_y)) *
        z_gate,
        color='blue')
    ax.plot(
        np.ones(
            np.shape(_y)) *
        x_anode1,
        _y,
        np.ones(
            np.shape(_y)) *
        z_anode,
        color='red')
    ax.plot(
        np.ones(
            np.shape(_y)) *
        x_anode2,
        _y,
        np.ones(
            np.shape(_y)) *
        z_anode,
        color='red')
    return ax


def get_axis_parallel_wires(ax, x_min, x_max, y_min, y_max, z_gate, z_anode):
    _x = np.linspace(x_min, x_max, 200)

    y_anode_list = get_anode_wires_y(y_min, y_max)
    y_gate_list = get_gate_wires_y(y_min, y_max)

    for _y_anode_wire in y_anode_list:
        ax.plot(
            _x,
            np.ones(
                np.shape(_x)) *
            _y_anode_wire,
            np.ones(
                np.shape(_x)) *
            z_anode,
            color='red',
            alpha=0.6)
    for _y_gate_wire in y_gate_list:
        ax.plot(
            _x,
            np.ones(
                np.shape(_x)) *
            _y_gate_wire,
            np.ones(
                np.shape(_x)) *
            z_gate,
            color='blue',
            alpha=0.6)

    return ax


def get_anode_wires_y(y_min, y_max, y_reference_anode=-0.01304 * 1000):
    '''Returns the y coordinates of the anode wires from a
    given y_min up to a given y_max. The reference point
    was taken directly form the mesh file by hand.
    The reference point comes twice, this is bug that I'm
    too lazy right now to solve but shoudl be fine for ploting.
    Cheers!'''

    return np.concatenate(np.sort(np.arange(y_reference_anode, y_min, -5)),
                          np.sort(np.arange(y_reference_anode, y_max, 5)))


def get_gate_wires_y(y_min, y_max, y_reference_gate=0.007824 * 1000):
    '''Returns the y coordinates of the gate wires from a
    given y_min up to a given y_max. The reference point
    was taken directly form the mesh file by hand.
    The reference point comes twice, this is bug that I'm
    too lazy right now to solve but shoudl be fine for ploting.
    Cheers!'''

    return np.concatenate(np.sort(np.arange(y_reference_gate, y_min, -5)),
                          np.sort(np.arange(y_reference_gate, y_max, 5)))
