# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 01:22:59 2021

@author: Duncan McArthur
"""

import numpy as np

__all__ = ['get_xi_from_theta','satCoords','distance','elevation','tMax']

###############################################################################

def get_xi_from_theta(theta,R,s,h0):
    """
    Get the orbit zenith-offset angle from the elevation angle

    Parameters
    ----------
    theta : float
        Elevation angle (rads).
    R : float
        Radius of the Earth (m).
    s : float
        Orbital altitude of satellite (m).
    h0 : float
        Altitude of receiver (m).

    Returns
    -------
    float
        Zenith-offset angle (rads).

    """
    # Global zenith angle (from centre of Earth)
    term  = np.sqrt((R + s)**2 - ((R + h0)*np.cos(theta))**2)
    numer = (R + h0)*np.cos(theta)**2 + np.sin(theta)*term
    denom = R + s
    return np.arccos(numer/denom)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def satCoords(Rs,wt,xi):
    """
    Returns the coordinates of the satellite in a Cartesian coordinate frame
    with origin at the centre of the Earth.

    Parameters
    ----------
    Rs : float
        Radial distance from the centre of the Earth to the satellite (R+s).
    wt : float
        Orbital phase of satellite (omega * t).
    xi : float
        Angle between the satellite and zenith position wrt centre of the Earth.

    Returns
    -------
    float, array-like
        Cartesian coordinate array for location of satellite.

    """
    return Rs * np.array([np.sin(xi) * np.cos(wt), np.sin(wt), np.cos(xi)*np.cos(wt)])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def distance(Rs,wt,xi,OGScoords):
    """
    Calculate the distance between the satellite and OGS given by the norm of
    the difference between their position vectors.

    Parameters
    ----------
    Rs : float
        Radial distance from the centre of the Earth to the satellite (R+s).
    wt : float
        Orbital phase of satellite (omega * t).
    xi : float
        Angle between the satellite and zenith position wrt centre of the Earth.
    OGScoords : float, array-like
        Position vector of the OGS in a Cartesian coordinate frame centered on 
        the Earth.

    Returns
    -------
    float
        Distance between OGS and satellite.

    """
    SATcoords = satCoords(Rs,wt,xi) # Satellite coordinates in Cartesian
    return np.linalg.norm(SATcoords - OGScoords)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def elevation(Rs,wt,xi,OGScoords):
    """
    Returns the elevation angle of the satellite wrt the OGS (rads).

    Parameters
    ----------
    Rs : float
        Radial distance from the centre of the Earth to the satellite (R+s).
    wt : float
        Orbital phase of satellite (omega * t).
    xi : float
        Angle between the satellite and zenith position wrt centre of the Earth.
    OGScoords : float, array-like
        Position vector of the OGS in a Cartesian coordinate frame centered on 
        the Earth.

    Returns
    -------
    float
        Elevation angle of satellite from OGS (rads).

    """
    SATcoords = satCoords(Rs,wt,xi)  # Satellite coordinates in Cartesian
    L = distance(Rs,wt,xi,OGScoords) # Distance between OGS and satellite
    return np.arcsin((SATcoords[2] - OGScoords[2]) / L)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def tMax(R,s,h0,omega,xi):
    """
    Calculate the maximum transmission time window half-width (dt) for a given
    satellite orbit.

    Parameters
    ----------
    R : float
        Radius of the Earth.
    s : float
        Altitude of satellite above Earths surface.
    h0 : float
        Altitude of OGS above sea-level    
    omega : float
        Angular velocity of staellite.
    xi : float
        Off-zenith angle wrt OGS for satellite orbit.

    Returns
    -------
    float
        Max transmission time window half-width.

    """
    return np.arccos((R + h0) / ((R + s)*np.cos(xi))) / omega