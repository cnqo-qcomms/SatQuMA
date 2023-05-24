# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 00:25:28 2021

@author: Duncan McArthur
"""

#from os.path import join

import numpy as np

__all__ = ['get_f_atm','make_f_atm','default_datafile']

###############################################################################

def make_f_atm(datafile,wl):
    """
    Make an atmospheric transmission efficiency (loss) function by 
    interpolating MODTRAN data.

    Parameters
    ----------
    datafile : string
        Name of MODTRAN data path/file.
    wl : float
        The specific wavelength (nm).

    Returns
    -------
    f_atm : function
        Atmospheric transmissvity function.

    """
    # Read the data file header
    try:
        with open(datafile,'rt') as fp:
            header = fp.readline().strip().replace('#','').replace(' nm','')
            cols   = header.split(',') # Split input line by commas
    except:
        raise FileNotFoundError(datafile)
    # Find wavelength data column from header 
    iwl  = cols.index(str(int(wl)))
    
    # Read in data as an array excluding the header
    data = np.loadtxt(datafile,skiprows=1,usecols=(0,iwl,),delimiter=',')

    # Generate interpolated MODTRAN atmospheric data
    from scipy.interpolate import interp1d
    # Generate interpolation function
    f_atm = interp1d(np.radians(data[:,0]), data[:,1])
    return f_atm

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def make_f_atm_local(wl):
    """
    Make an atmospheric transmission efficiency (loss) function by 
    interpolating MODTRAN data from a local file.
        \lambda \in[785,850] nm, in steps of 5 nm.

    Parameters
    ----------
    wl : float
        The specific wavelength (nm).

    Returns
    -------
    f_atm : function
        Atmospheric transmissvity function.

    """
    from .MODTRAN_data import get_atm_data
    elev, loss = get_atm_data(wl) # Get data from local file
    # Generate interpolated MODTRAN atmospheric data
    from scipy.interpolate import interp1d
    f_atm = interp1d(np.radians(elev), loss)
    return f_atm

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def default_datafile():
    return 'MODTRAN_wl_785.0-850.0-5.0nm_h1_500.0km_h0_0.0km_elevation_data.csv'

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_f_atm(loss_params):
    """
    Returns an atmospheric transmissivity function with an elevation angle
    dependence. Returns a dummy function when not needed.

    Parameters
    ----------
    loss_params : dict
        Dictionary of loss parameters.

    Returns
    -------
    f_atm : function
        Atmospheric transmissivity angular function

    """
    if loss_params['tReadLoss']:
        # Function not required so use dummy
        f_atm = lambda *args, **kwargs: None
    else:
        # Define the data file containing atmospheric data
        if loss_params['atm_file'] == '':
            #datafile = default_datafile()
            # Try and use local stored data arrays
            f_atm = make_f_atm_local(loss_params['wvl'])
        else:
            datafile = loss_params['atm_file']
            f_atm = make_f_atm(datafile,loss_params['wvl'])
        #f_atm = make_f_atm(datafile,loss_params['wvl'])        
    return f_atm