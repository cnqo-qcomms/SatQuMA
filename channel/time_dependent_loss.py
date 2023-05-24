# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 00:13:57 2021

@author: Duncan McArthur
"""

from os.path import join

import numpy as np

from .diffraction.diffraction import diffract
from .orbital.circular_polarorbit import (elevation, distance, tMax, 
                                          get_xi_from_theta)
from .atmosphere.atmos_data import (make_f_atm, default_datafile)

__all__ = ['interp_atm_data','time_dependent_losses']

#############################################################################

def interp_atm_data(wl,datafile=None):
    """
    Generate an elevation dependent atmospheric transmissivity function by 
    interpolating data from a file.

    Parameters
    ----------
    wl : float
        Wavelength (nm).
    datafile : str, optional
        Path/file. The default is None.

    Returns
    -------
    f_atm : function
        Elevation dependent transmissivity function.

    """
    if datafile is None:
        # Use the default data file
        #datafile = atm.default_datafile()
        datafile = default_datafile()
    # Make an interpolated function
    #f_atm = atm.make_f_atm(datafile,wl)
    f_atm = make_f_atm(datafile,wl)
    return f_atm

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def time_dependent_losses(R,hsat,xi,hOGS,wl,aT,aR,w0,f_atm,eta_int):
    """
    Generate time dependent loss array based on diffraction, atmospheric and
    inherent system losses.

    Parameters
    ----------
    R : float
        Radius of the Earth (m).
    hsat : float
        Orbital altitude of satellite (m).
    xi : float
        Orbit tilt (from zenith) angle (rads).
    hOGS : float
        Altitude of receiver (m).
    wl : float
        Transmission wavelength (nm).
    aT : float
        Transmission aperture radius (m).
    aR : float
        Receiver aperture radius (m).
    w0 : float
        Beam waist at focus (m).
    f_atm : function
        Atmospheric loss function.
    eta_int : float
        Constant system losses as efficiency.

    Returns
    -------
    vals : float
        Loss versus time(elevation) array. 

    """
    # Constants  
    G           = 6.67430e-11    # Gravitational constant (m^3 kg^-1 s^-1)
    M           = 5.9724e24      # Mass of Earth (kg)
    # System parameters
    k           = 2*np.pi / (wl*10**(-9))      # Angular wavenumber
    omega       = np.sqrt(G*M / (R + hsat)**3) # Angular velocity of satellite
    OGScoords   = np.array([0,0,R + hOGS])     # Define receiver at North pole

    # Initialise arrays
    #tmax      = int(orb.tMax(R,hsat,hOGS,omega,xi)) # Max transmission window half-width
    tmax      = int(tMax(R,hsat,hOGS,omega,xi)) # Max transmission window half-width
    vals      = np.empty((2*tmax+1,7))          # Initisalise array to store values
    count     = 0                               # Initialise time-slot counter
    for t in range(tmax,-tmax-1,-1):
        #elev      = orb.elevation(R+hsat,omega*t,xi,OGScoords) # Elevation angle (rads) 
        #L         = orb.distance(R+hsat,omega*t,xi,OGScoords)  # Sat-to-OGS range
        elev      = elevation(R+hsat,omega*t,xi,OGScoords) # Elevation angle (rads) 
        L         = distance(R+hsat,omega*t,xi,OGScoords)  # Sat-to-OGS range
    
        # Calculate abs&scat for current system by interpolating data
        eta_atm   = f_atm(elev)
    
        # Calculate diffraction for current system (Fraunhoffer)
        eta_diff   = diffract(aT,aR,L,k,w0)
        
        # Calculate total efficiency of current system
        eta_tot    = eta_atm * eta_diff * eta_int
        
        # Store data
        vals[count,0] = t          # Orbital time of current system (s)
        vals[count,1] = elev       # Elevation angle (degs)
        vals[count,2] = eta_tot    # Total efficiency for current system
        vals[count,3] = eta_diff   # Diffraction efficiency for current system
        vals[count,4] = eta_atm    # Atmospheric transmissivity for current system
        vals[count,5] = eta_int    # Intrinsic system losses
        vals[count,6] = L          # Range (m)
        count += 1
    # Header for loss files
    header  = 'Time (s),Elevation (rad),eta_tot,eta_diff,eta_atm,eta_sys,Distance (m)'
    return vals, header

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_losses(theta_max,loss_params,f_atm,tPrint,outpath):
    """
    Returns the loss array by either generating data or reading from file.

    Parameters
    ----------
    theta_max : float
        Max elevation of orbit (rad).
    loss_params : dict
        Dictionary of loss related parameters.
    f_atm : function
        Elevation dependent atmospheric transmissivity function.
    tPrint : bool
        Flag for printing output to screen.
    outpath : str
        Path to write loss file to (if requsted).

    Returns
    -------
    loss_data : float, array-like
        Array containing time, elevation and transmission efficiency data.

    """
    if loss_params['tReadLoss']:
        # File containing loss data (for given theta_max value below)
        loss_file = loss_params['loss_file'].format(theta_max)
    
        if tPrint:
            print('Reading losses from file:',join(loss_params['loss_path'],loss_file))
            print('-'*60)
    
        #***********************************************************************
        #***********************************************************************
        # Read in free-space loss data from CSV file
        #***********************************************************************
        #***********************************************************************
        # Read from a local file and skip first line and take data only from the 
        # specified column. The free space loss should be arranged in terms of
        # time slots, t.
        loss_data = np.loadtxt(join(loss_params['loss_path'],loss_file),
                               delimiter=',',skiprows=1,
                               usecols=(0,1,loss_params['loss_col']-1,))
    else:
        if tPrint:
            print('Generating losses for theta_max = {} deg'.format(np.degrees(theta_max)))
            print('-'*60)
            
        # Calculate the angle between the orbital plane and the zenith plane
        xi = get_xi_from_theta(theta_max,loss_params['R_E'],loss_params['h_T'],loss_params['h_R'])
        #***********************************************************************
        #***********************************************************************
        # Generate free-space loss data
        #***********************************************************************
        #***********************************************************************
        # The free space loss should be arranged in terms of time slots, t.
        loss_data, loss_head = \
            time_dependent_losses(loss_params['R_E'],loss_params['h_T'],xi,
                                  loss_params['h_R'],loss_params['wvl'],
                                  loss_params['aT'],loss_params['aR'],
                                  loss_params['w0'],f_atm,loss_params['eta_int'])
        if loss_params['tWriteLoss']:
            loss_file = 'FS_loss_th_m_{:5.2f}_wl_{:.0f}nm_h_{}km_h1_{}km_aT_{}m_aR_{}m_w0_{}m.csv'.format(
                    np.degrees(theta_max),loss_params['wvl'],
                    loss_params['h_T']/1e3,loss_params['h_R']/1e3,
                    loss_params['aT'],loss_params['aR'],loss_params['w0'])
            if tPrint:
                print('Saving losses to file:',join(outpath,loss_file))
                print('-'*60)
            np.savetxt(join(outpath,loss_file),loss_data,delimiter=',',header=loss_head)
    return loss_data