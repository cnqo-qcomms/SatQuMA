# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 02:05:57 2021

@author: Duncan McArthur
"""

from sys import float_info
# Extract the smallest float that the current system can:
num_min = float_info.epsilon # round (relative error due to rounding)
# Extract the largest float that the current system can represent
num_max = float_info.max

import math
import numpy as np

from ..maths import (h, heaviside, gamma)
from ..error_correction.functions import (logM)
from .init_efficient_BB84 import *
from .func_efficient_BB84 import *

__all__ = ['set_params','key_length', 'key_length_inv']

###############################################################################

def set_params(Pec,QBERI,ls,dt,mu3,time0pos,Pap,FSeff,Npulse,boundFunc, 
               eps_c,eps_s,num_zero,errcorrFunc,fEC,NoPass):
    """
    Wrap a dictionary of arguments required by key_length functions in a tuple
    for use with scipy.optimize

    Parameters
    ----------
    mu3 : int
        Intensity of pulse 3.
    ls : float
        Excess loss (dB).
    dt : int
        Transmission time window half-width (s).
    time0pos : int
        Index of t = 0 point in transmission window/array.
    Pec : float
        Probability of extraneous counts.
    QBERI : float
        Intrinsic quantum bit error rate.
    Pap : float
        Probability of after-pulse event.
    FSeff : float, array-like
        Free space transmissivity.
    Npulse : int
        Number of pulses sent.
    boundFunc : str
        Name of tail bounds to use.
    eps_c : float
        Correctness parameter.
    eps_s : float
        Secrecy parameter.
    num_zero : float
        Value to use when denominators approach zero.
    errcorrFunc : str
        Name of error correction method.
    fEC : float
        Error correction factor (> 1.0)
    NoPass : int
        Number of satellite overpasses.

    Returns
    -------
    tuple
        Dictionary of arguments wrapped in a tuple.

    """
    arg_dict = dict() # Initialise dictionary
    # Store parameters
    arg_dict['mu3']         = mu3
    arg_dict['ls']          = ls
    arg_dict['dt']          = dt
    arg_dict['time0pos']    = time0pos
    arg_dict['Pec']         = Pec
    arg_dict['QBERI']       = QBERI
    arg_dict['Pap']         = Pap
    arg_dict['FSeff']       = FSeff
    arg_dict['Npulse']      = Npulse
    arg_dict['boundFunc']   = boundFunc
    arg_dict['eps_c']       = eps_c
    arg_dict['eps_s']       = eps_s
    arg_dict['num_zero']    = num_zero
    arg_dict['errcorrFunc'] = errcorrFunc
    arg_dict['fEC']         = fEC
    arg_dict['NoPass']      = NoPass
    return arg_dict

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_params(args):
    """
    Returns arguments needed by the key_length function from a dictionary
    stored within a tuple.

    Parameters
    ----------
    args : tuple
        Arguments passed to key_length functions.

    Returns
    -------
    mu3 : int
        Intensity of pulse 3.
    ls : float
        Excess loss (dB).
    dt : int
        Transmission time window half-width (s).
    time0pos : int
        Index of t = 0 point in transmission window/array.
    Pec : float
        Probability of extraneous counts.
    QBERI : float
        Intrinsic quantum bit error rate.
    Pap : float
        Probability of after-pulse event.
    FSeff : float, array-like
        Free space transmissivity.
    Npulse : int
        Number of pulses sent.
    boundFunc : str
        Name of tail bounds to use.
    eps_c : float
        Correctness parameter.
    eps_s : float
        Secrecy parameter.
    num_zero : float
        Value to use when denominators approach zero.
    errcorrFunc : str
        Name of error correction method.
    NoPass : int
        Number of satellite overpasses.

    """
    mu3         = args['mu3']
    ls          = args['ls']
    dt          = args['dt']
    time0pos    = args['time0pos']
    Pec         = args['Pec']
    QBERI       = args['QBERI']
    Pap         = args['Pap']
    FSeff       = args['FSeff']
    Npulse      = args['Npulse']
    boundFunc   = args['boundFunc']
    eps_c       = args['eps_c']
    eps_s       = args['eps_s']
    num_zero    = args['num_zero']
    errcorrFunc = args['errcorrFunc']
    fEC         = args['fEC']
    NoPass      = args['NoPass']
    return mu3, ls, dt, time0pos, Pec, QBERI, Pap, FSeff, Npulse, boundFunc, \
            eps_c, eps_s,num_zero, errcorrFunc, fEC, NoPass


#******************************************************************************
#******************************************************************************
# Function(s) to calculate the secure key length
#******************************************************************************
#******************************************************************************

# Secure Key Length (SKL) function - for direct calculation
def key_length(x, args):
    """
    Returns the secure key length for an asymmetric BB84 protocol with weak
    coherent pulses and 2 'decoy' states. The intensity of the weak coherent
    pulse 3, mu_3, is assumed to be a pre-defined global parameter.
    Final expression is Eq. (1) in [1].

    Parameters
    ----------
    x : float, array/tuple
        x[0] = Asymmetric basis choice probability - Px.
        x[1] = Weak coherent pulse 1 probability - pk_1
        x[2] = Weak coherent pulse 2 probability - pk_2
        x[3] = Weak coherent pulse 1 intensity - mu_1
        x[4] = Weak coherent pulse 1 intensity - mu_2

    Returns
    -------
    l : float
        Secure key length (in bits).
    QBERx : float
        The Quantum Bit Error rate for the X basis.
    phi_x : float
        The phase error rate for the X basis.
    nX : float
        The total number of measured events in the X basis.
    nZ : float
        The total number of measured events in the Z basis.
    mXtot : float
        The total number of measurement errors in the X basis.
    lambdaEC : float
        The estimated number of bits used for error correction.
    sx0 : float
        The number of vacuum events in the X basis.
    sx1 : float
        The number of single-photon events in the X basis.
    vz1 : float
        The number of bit errors associated with single-photon events in the Z 
        basis.
    sz1 : float
        The number of single-photon events in the Z basis.
    mpn : float
        The mean photon number of the signal.
    
    """
    ###########################################################################
    ## Get fixed parameters from input dictionary
    ###########################################################################
    mu3, ls, dt, time0pos, Pec, QBERI, Pap, FSeff, Npulse, boundFunc, \
            eps_c, eps_s, num_zero, errcorrFunc, fEC, NoPass = get_params(args)
                
    ###########################################################################
    ## Store input as named arrays/parameters for ease of use
    ###########################################################################
    # Asymmetric basis choice probability
    Px = x[0]
    # Weak coherent pulse intensities, mu3 is fixed
    mu = np.array([x[3],x[4],mu3])
    # Probability of Alice sending a pulse of intensity mu_k
    P  = np.array([x[1],x[2],1 - x[1] - x[2]])
    
    ###########################################################################
    ## Define total system loss and convert to efficiency
    ###########################################################################
    # Excess loss in decibels
    etaExcess = 10**(-ls / 10.0)
    # Scale excess loss by detector efficiency
    eta = etaExcess #*eta_d
    # Take the outer product of FSeff, the free space loss, and mu, the pulse
    # intensities, where FSeff has a length t.
    # Use dt to take a window of t values from FSeff.
    # Returns the (3,t) array mu_FSeff
    # Symmetric transmsisison window about time0pos
    mu_FSeff = np.outer(mu, FSeff[time0pos-dt:time0pos+dt+1])

    # Exponential loss decay function
    # Returns the (3,t) array exp_loss_jt
    exp_loss_jt = np.exp(-mu_FSeff*eta)

    ###########################################################################
    ## Estimate count and error rates for signals
    ###########################################################################
    # Expected detection rate (including afterpulse contributions)
    # Returns the (3,t) array Dj
    Dj = DRate_j(eta,Pap,Pec,exp_loss_jt)
    ###print('Dj    =',np.sum(Dj,axis=1))
    # Probability of having a bit error per intensity for a given time slot
    # Returns the (3,t) array ej
    ej = error_j(Dj,Pap,Pec,QBERI,exp_loss_jt)
    
    ###########################################################################
    ## Store some useful array products
    ###########################################################################
    # Take the dot product of the (3,) array P and the (3,t) array ej.
    # Returns the (t,) array P_dot_ej
    P_dot_ej   = np.dot(P, ej)
    # Take the dot product of the (3,) array P and the (3,t) array Dj.
    # Returns the (t,) array P_dot_Dj
    P_dot_Dj   = np.dot(P, Dj)
    # Do an element-wise multiplication of the (3,) array P and the (3,t) 
    # array Dj.
    # Returns the (3,t) array P_times_Dj
    P_times_Dj = np.transpose(np.multiply(np.transpose(Dj), P))

    ###########################################################################
    ## Estimate count statistics
    ###########################################################################
    # Number of events in the sifted X basis for each time slot and intensity
    # Returns the (3,t) array nx
    nx = nxz(Px, Px, Npulse, P_times_Dj)
    # Number of events in the sifted Z basis for each time slot and intensity
    # Returns the (3,t) array nz
    nz = nxz(1 - Px, 1 - Px, Npulse, P_times_Dj)

    # Total number of events in the sifted X basis for each intensity
    # Returns the (3,) array nx_mu
    nx_mu = np.sum(nx, axis=1)
    # Total number of events in the sifted Z basis for each intensity
    # Returns the (3,) array nz_mu
    nz_mu = np.sum(nz, axis=1)
    
    # Total number of events in the sifted X basis
    nX = np.sum(nx_mu)
    # Total number of events in the sifted Z basis
    nZ = np.sum(nz_mu) # Not used but written out
    
    # Number of errors in the sifted X basis for each time slot
    # Returns the (t,) array mX
    mX = mXZ(nx, P_dot_Dj, P_dot_ej)
    mZ = mXZ(nz, P_dot_Dj, P_dot_ej)
    # Number of errors in the sifted X basis for each time slot and intensity
    # Returns the (3,t) array mx
    #mx = mxz(mX, P_dot_Dj, P_times_Dj) # Not used
    mz = mxz(mZ, P_dot_Dj, P_times_Dj)
    # Total number of errors in the sifted X basis
    mXtot = np.sum(mX)
    mZtot = np.sum(mZ)
    # Number of errors in the sifted X basis for each intensity
    # Returns the (3,) array mXj
    #mXj = np.sum(mx, axis=1) # Not used
    mZj = np.sum(mz, axis=1)
    
    ###########################################################################
    ## Estimate bounds on count estimates due to statistical fluctuations
    ###########################################################################
    # Upper and lower bounds used to estimate the ideal number of X and Z basis
    # events accounting for statistical fluctuations  
    if boundFunc in ['Chernoff','chernoff']:
        # Use Chernoff bounds
        # Returns the (3,) arrays nXmin, nXplus
        nXmin, nXplus = nXZpm(mu,P,nx_mu,eps_s)
        # Returns the (3,) arrays nZmin, nZplus
        nZmin, nZplus = nXZpm(mu,P,nz_mu,eps_s)
        # Returns the (3,) arrays mXmin and mXplus
        #mXmin, mXplus = mXZpm(mu,P,mXj,eps_s) # Not used
        # Returns the (3,) arrays mZmin and mZplus
        mZmin, mZplus = mXZpm(mu,P,mZj,eps_s)
    elif boundFunc in ['Hoeffding','hoeffding']:
        # Use Hoeffding bound
        # Returns the (3,) arrays nXmin, nXplus
        nXmin, nXplus = nXZpm_HB(mu,P,nx_mu,nX,eps_s)
        # Returns the (3,) arrays nZmin, nZplus
        nZmin, nZplus = nXZpm_HB(mu,P,nz_mu,nZ,eps_s)
        # Returns the (3,) arrays mXmin and mXplus
        #mXmin, mXplus = mXZpm_HB(mu,P,mXj,mXtot,eps_s) # Not used
        # Returns the (3,) arrays mZmin and mZplus
        mZmin, mZplus = mXZpm_HB(mu,P,mZj,mZtot,eps_s)
    elif boundFunc in ['Asymptotic','asymptotic']:
        # Use asymptotic bounds - no bounds
        # Returns the (3,) arrays nXmin, nXplus
        nXmin, nXplus = nXZpm_inf(mu,P,nx_mu)
        # Returns the (3,) arrays nZmin, nZplus
        nZmin, nZplus = nXZpm_inf(mu,P,nz_mu)
        # Returns the (3,) arrays mXmin and mXplus
        #mXmin, mXplus = mXZpm_inf(mu,P,mXj,eps_s) # Not used
        # Returns the (3,) arrays mZmin and mZplus
        mZmin, mZplus = mXZpm_inf(mu,P,mZj)
    else:
        # Use Chernoff bounds
        # Returns the (3,) arrays nXmin, nXplus
        nXmin, nXplus = nXZpm(mu,P,nx_mu,eps_s)
        # Returns the (3,) arrays nZmin, nZplus
        nZmin, nZplus = nXZpm(mu,P,nz_mu,eps_s)
        # Returns the (3,) arrays mXmin and mXplus
        #mXmin, mXplus = mXZpm(mu,P,mXj,eps_s) # Not used
        # Returns the (3,) arrays mZmin and mZplus
        mZmin, mZplus = mXZpm(mu,P,mZj,eps_s)

    ###########################################################################
    ## Calculate the number of n-photon events
    ###########################################################################
    # Number of vacuum events in the sifted X basis
    sx0 = max(s0(mu,P,nXmin), num_zero)
    # Number of vacuum events in the sifted Z basis
    sz0 = max(s0(mu,P,nZmin), num_zero)

    # Number of single photon events in the sifted X basis
    sx1 = max(s1(mu,P,nXmin,nXplus,sx0), num_zero)
    # Number of single photon events in the sifted Z basis
    sz1 = max(s1(mu,P,nZmin,nZplus,sz0), num_zero)

    ###########################################################################
    ## Calculate metrics such as the error rate, QBER and mean photon number
    ###########################################################################
    # Number of bit errors associated with single photon events in the sifted
    # Z basis 
    vz1   = min(max(vxz1(mu,P,mZmin,mZplus), num_zero), mZtot)
    # Ratio between the number of bit errors and the number of events
    # associated with single photons in the sifted Z basis.
    ratio = min(vz1 / sz1, 1 - num_min)
    # The quantum bit error rate in the sifted X basis
    QBERx = mXtot / nX
    # Calculate the mean photon number
    mpn   = mean_photon_a(P,mu)

    ###########################################################################
    ## Estimate the number of bits sacrificed for error correction
    ###########################################################################
    if errcorrFunc in ['logM','logm']:
        # Tomamichael bound - correct for lower bound on fEC for large blocks
        lambdaEC = max(logM(nX, QBERx, eps_c),fEC*nX*h(QBERx))
    elif errcorrFunc in ['Block','block']:
        lambdaEC = fEC * nX * h(QBERx)
    elif errcorrFunc in ['mXtot','mxtot']:
        lambdaEC = fEC * mXtot
    elif errcorrFunc in ['None','none']:
        lambdaEC = 0
    else:
        lambdaEC = 0
    
    ###########################################################################
    ## Calculate the approximate length of the secret key per pass
    ## See Eq. (1) in [1].
    ###########################################################################
    if boundFunc in ['Asymptotic','asymptotic']:
        # Asymptotic phase error rate for single photon events in the sifted X
        # basis
        phi_x = max(0, min(ratio, 0.5))
        if phi_x < 0 or phi_x > 1:
            print('phi_x = {}'.format(phi_x))
        # Secret key length in the asymptotic regime
        l = max((sx0 + sx1 * (1 - h(phi_x)) - lambdaEC) * 
             heaviside(mu[0] - mu[1] - mu[2]) * heaviside(P[2]), 0.0)
    else:
        # Phase error rate for single photon events in the sifted X basis
        phi_x = min(ratio + gamma(eps_s,ratio,sz1,sx1), 0.5)
        # Secret key length in the finite regime
        l = max((sx0 + sx1 * (1 - h(phi_x)) - lambdaEC -
             6*math.log(21.0 / eps_s, 2) - math.log(2.0 / eps_c, 2)) * 
             heaviside(mu[0] - mu[1] - mu[2]) * heaviside(P[2]), 0.0)
        l = l / NoPass # Normalise by the number of satellite passes used

    return l, QBERx, phi_x, nX, nZ, mXtot, lambdaEC, sx0, sx1, vz1, sz1, mpn

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Secure Key Length (SKL) inverse function - returns (1 / SKL) - for optimizer
def key_length_inv(x,args):
    """
    Returns the inverse of the secure key length for an asymmetric BB84 
    protocol with weak coherent pulses and 2 'decoy' states. The intensity of 
    the weak coherent pulse 3, mu_3, is assumed to be a pre-defined global 
    parameter.
    Final expression is Eq. (1) in [1].

    Parameters
    ----------
    x : float, array/tuple
        x[0] = Asymmetric basis choice probability - Px.
        x[1] = Weak coherent pulse 1 probability - pk_1
        x[2] = Weak coherent pulse 2 probability - pk_2
        x[3] = Weak coherent pulse 1 intensity - mu_1
        x[4] = Weak coherent pulse 1 intensity - mu_2

    Returns
    -------
    1/l : float
        Inverse of the secure key length (in bits).

    """
    # Safety check that all parameters are positive
    if (np.any(x[x < 0])):
        return num_max  # ~1/0
    #print(args)
    # Safety check that the parameters satisfy the constraints
    C = bool_constraints(x[0],x[1],x[2],x[3],x[4],args['mu3'])
    if (not np.all(C)):
        return num_max  # ~1/0
    
    # Calculate the secure key length (normalised by NoPass)
    l, _, _, _, _, _, _, _, _, _, _, _ = key_length(x,args)

    if (l > 0):
        return (1.0/l)  # Inverse key length
    elif (l == 0):
        return num_max  # ~1/0
    else:
        #return num_max  # ~1/0
        print("Warning! Unexpected key length:", l)
        #return l        # Negative key length, NaN?
        return num_max  # ~1/0