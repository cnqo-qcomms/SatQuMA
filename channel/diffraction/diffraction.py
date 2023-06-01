# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 00:14:35 2021

@author: Duncan McArthur
"""

from scipy.integrate import quad
from scipy.special import j0
import numpy as np

__all__ = ['diffract']

###############################################################################

def GaussianMode(r,w0):
    """
    Calculates a power-normalised Guassian field distribution at focus.

    Parameters
    ----------
    r : float
        Cylindrical radius.
    w0 : float
        Beamw waist.

    Returns
    -------
    float
        Gaussian amplitude.

    """
    return np.math.sqrt(2/np.pi) * np.exp(-(r/w0)**2) / w0

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def Pff_integrand(r,rho,z,k,w0):
    """
    Integrand to (partially) numerically calculate the far-field intensity of a 
    Gaussian field through an aperture using the Fraunhoffer approximation. 
    This integrand considers light originating from a single point in the 
    transmitter plane.
    Missing factor 2 pi from azimuthal integral.

    Parameters
    ----------
    r : float
        Transverse radius in receiver plane.
    rho : float
        Transverse radius in transmitter plane.
    z : float
        Distance between transmit-receive planes.
    k : float
        Angular wavenumber.
    w0 : float
        Beam waist at focus.

    Returns
    -------
    float
        Integrand for received intensity from a single point.

    """
    return r*GaussianMode(r,w0)*j0(k*rho*r/z)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def Pr_integrand(rho,z,aT,k,w0):
    """
    Integrand to calculate the power of a Gaussian field through an aperture
    at the receiver plane.
    Missing factor 2 pi from azimuthal integral.

    Parameters
    ----------
    rho : float
        Transverse radius in transmitter plane.
    z : float
        Distance between transmit-receive planes.
    aT : float
        Transmitter aperture radius.
    k : float
        Angular wavenumber.
    w0 : float
        Beam waist at focus.

    Returns
    -------
    float
        Integrand.

    """
    int1, _ = quad(Pff_integrand, 0, aT, args=(rho,z,k,w0))
    return (k/z)**2 * rho * abs(int1)**2

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def Pt_integrand(r,w0):
    """
    Integrand to calculate the power of a Gaussian field through an aperture
    at the transmitter plane.
    Missing factor 2 pi from azimuthal integral.

    Parameters
    ----------
    r : float
        Transverse radius in transmitter plane.
    w0 : float
        Beam waist at focus.

    Returns
    -------
    float
        Integrand.

    """
    return r * abs(GaussianMode(r,w0))**2

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def diffract(aT,aR,z,k,w0):
    """
    Return the recieved to transmsitted power ratio. Calculated using the
    Fraunhoffer approximation to the Sommerfeld-Rayliegh integral

    Parameters
    ----------
    aT : float
        Tranmsit aperture radius (m).
    aR : float
        Receive aperture radius (m).
    z : float
        Propagation length (m).
    k : float
        Angular wavenumber (1/m).
    w0 : float
        Beam waist (1/e radius) at focus (m).

    Returns
    -------
    float
        Transmit/receive power ratio.

    """
    Pr, er = quad(Pr_integrand, 0, aR, args=(z,aT,k,w0))
    Pt, et = quad(Pt_integrand, 0, aT, args=(w0,))
    return Pr/Pt