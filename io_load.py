import root_numpy as rootnp
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import ROOT


def load_root(filepath):
    '''Loads the .root EField computed map, output of KEMField simulations,
    into a pandas dataframe. Each row is a field point. x, y, z are collumns,
    leading to Nx*Ny*Nz number of entries.''' 

    # Load file
    file = ROOT.TFile(filepath,'read')

    tree = file.Get('t1')

    entries = tree.GetEntriesFast()

    print('Read %d entries in file' %entries)

    treedata= rootnp.tree2array(tree, branches=['evtnb',
                                            'Phi',
                                            'Ex',
                                            'Ey',
                                            'Ez',
                                            'ExN',
                                            'EyN',
                                            'EzN',
                                            'x',
                                            'y',
                                            'z'])
    EFpoints = pd.DataFrame(treedata)
    
    # Convert to mm and compute E
    EFpoints['x'] = EFpoints['x']*1000
    EFpoints['y'] = EFpoints['y']*1000
    EFpoints['z'] = EFpoints['z']*1000
    
    #Calculate r2 and |E|
    EFpoints['r'] = np.sqrt(np.power(EFpoints.x,2) + np.power(EFpoints.y,2))
    EFpoints_cut = EFpoints[EFpoints.r < 720]
    print('Cut %d (%.2f%%) useless points.' %(len(EFpoints)-len(EFpoints_cut),
                                            (len(EFpoints)-len(EFpoints_cut))/len(EFpoints)*100))
    return EFpoints_cut


def load_several(ID_list,path, pre_name):
    '''Loads computed EField .root files from name and ID and returns a 
    concatenated pd.DataFrame.'''
    dfs = []
    path_cache = path + 'cache/'
    path_files = path + 'solved_outputs/EField_SR0/'
    for _ID in ID_list:
        _filepath = path_files + pre_name + '%d.root'%_ID
        try:
            _df = load_root(_filepath)
            dfs.append(_df)
            print('Loaded file with ID: %d' %_ID)
        except:
            print('Could not load file with ID: %d (%s)' %(_ID,_filepath))
    _data = pd.concat(dfs)
    _data.to_pickle(path_cache + 'dfsave_%s.pkl' %(datetime.utcnow().strftime("%d%m%Y_%H%M")))
    return _data
