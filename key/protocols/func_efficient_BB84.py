# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 01:56:35 2021

@author: Duncan McArthur
"""

import numpy as np

__all__ = ['DRate_j','error_j','nxz','mXZ','mxz','nXZpm','nXZpm_HB','nXZpm_inf',
           'tau','s0','s1','mXZpm','mXZpm_HB','mXZpm_inf','vxz1',
           'mean_photon_a','mean_photon_v']

###############################################################################

def DRate_j(eta,Pap,Pec,exp_loss_jt):
    """
    Calculates the expected detection rate including afterpulse contributions
    for each intensity and time slot.
    Defined as R_k in Sec. IV of [1].

    Parameters
    ----------
    eta : float
        Excess loss parameter.
    Pap : float
        Probability of an afterpulse event.
    Pec : float
        Extraneous count probability.
    exp_loss_jt : float, array
        Loss, per intensity per time slot, decay function.

    Returns
    -------
    float, array
        Expected detection rate.

    """
    return (1 + Pap)*(1 - (1 - 2*Pec)*exp_loss_jt)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def error_j(Dj,Pap,Pec,QBERI,exp_loss_jt):
    """
    Calculates the conditional probability for a pulse of intensity mu_j
    to cause an error, after sifting, in the time slot t.
    Defined as e_k in Sec. IV of [1].

    Parameters
    ----------
    Dj : float, array
        Expected detection rate.
    Pap : float
        Probability of an afterpulse event.
    Pec : float
        Extraneous count probability.
    QBERI : float
        Intrinsic Quantum Bit Error Rate.
    exp_loss_jt : float, array
        Loss, per intensity per time slot, decay function.

    Returns
    -------
    float, array
        Error rate per intensity per time slot.

    """
    return Pec + (0.5*Pap*Dj) + QBERI*(1 - exp_loss_jt)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def nxz(PAxz,PBxz,Npulse,P_times_Dj):
    """
    Calculates the number of events in the X or Z sifted basis per pulse
    intensity per time slot.
        nx[j,t] or nz[j,t];  j = {1:3}, t = {1:Nt}

    Parameters
    ----------
    PAxz : float
        Probability of Alice preparing a state in the X/Z basis.
    PBxz : float
        Probability of Bob measuring a state in the X/Z basis.
    Npulse : integer/float
        Number of pulses sent by Alice.
    P_times_Dj : float, array
        The element-wise multiplication of the intensity probability array P
        with the expected detection rate per time slot array Dj.

    Returns
    -------
    float, array
        The number of events in the sifted X/Z basis.

    """
    return PAxz*PBxz*P_times_Dj*Npulse

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def mXZ(nxz,P_dot_Dj,P_dot_ej):
    """
    Calculates the number of errors in the sifted X or Z basis for each time 
    slot.
        mX[t] or mZ[t];  t = {1:Nt}

    Parameters
    ----------
    nxz : float, array
        The number of events in the sifted X/Z basis per intensity per time
        slot.
    P_dot_Dj : float, array
        The dot product of the intensity probability array P with the expected 
        detection rate per time slot array Dj.
    P_dot_ej : float, array
        The dot product of the intensity probability array P with the 
        conditional probability for a pulse with a given intensity and time 
        slot to create an error array ej.

    Returns
    -------
    float, array
        The number of errors in the sifted X/Z basis per time slot.

    """
    return np.divide(np.multiply(P_dot_ej, np.sum(nxz, axis=0)), P_dot_Dj)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def mxz(mXZ,P_dot_Dj,P_times_Dj):
    """
    Calculates the number of errors in the sifted X or Z basis for a pulse
    with a given intensity in a particular time slot.
        mx[j,t] or mz[j,t];  j = {1:3}, t = {1:Nt}

    Parameters
    ----------
    mXZ : float, array
        The number of errors in the sifted X/Z basis per time slot.
    P_dot_Dj : float, array
        The dot product of the intensity probability array P with the expected 
        detection rate per time slot array Dj.
    P_times_Dj : float, array
        The element-wise multiplication of the intensity probability array P
        with the expected detection rate per time slot array Dj.

    Returns
    -------
    float, array
        Number of errors in sifted X/Z basis per intensity per time slot.

    """
    return np.divide(np.multiply(P_times_Dj, mXZ), P_dot_Dj)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def nXZpm(mu,P,nxz_mu,eps_s):
    """
    Calculates the upper and lower bounds on the number of events in the 
    sifted X or Z basis, for each pulse intensity using, the Chernoff bounds.
    Defined after Eq. (2) in [1].
        nXplus[j] and nXmin[j], or nZplus[j] and nZmin[j];  j = {1:3}

    Parameters
    ----------
    mu : float, array
        Pulse intensities.
    P : float, array
        Probability of Alice preparing a pulse intensity.
    nxz_mu : float, array
        Number of events per intensity for the sifted X/Z basis.
    eps_s : float
        The secrecy error; the key is eps_s secret.

    Returns
    -------
    nXZmin : float, array
        Lower bound on the expected number of events per intensity in the
        sifted X/Z basis.
    nXZplus : float, array
        Upper bound on the expected number of events per intensity in the
        sifted X/Z basis.

    """
    log_21es = np.math.log(21.0 / eps_s)
    term_m   = 0.5*log_21es + np.sqrt(2*nxz_mu*log_21es + 0.25*log_21es**2)
    term_p   = log_21es + np.sqrt(2*nxz_mu*log_21es + log_21es**2)

    nXZmin   = np.divide(np.multiply(np.exp(mu), nxz_mu - term_m), P)
    nXZplus  = np.divide(np.multiply(np.exp(mu), nxz_mu + term_p), P)
    return nXZmin, nXZplus

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def nXZpm_HB(mu,P,nxz_mu,nXZ,eps_s):
    """
    Calculates the upper and lower bounds on the number of events in the 
    sifted X or Z basis, for each pulse intensity using, the Hoeffding bound.
    Defined after Eq. (2) in [1].
        nXplus[j] and nXmin[j], or nZplus[j] and nZmin[j];  j = {1:3}

    Parameters
    ----------
    mu : float, array
        Pulse intensities.
    P : float, array
        Probability of Alice preparing a pulse intensity.
    nxz_mu : float, array
        Number of events per intensity for the sifted X/Z basis.
    nXZ : float, array
        Number of events per intensity per time slot for the sifted X/Z basis.
    eps_s : float
        The secrecy error; the key is eps_s secret.

    Returns
    -------
    nXZmin : float, array
        Lower bound on the expected number of events per intensity in the
        sifted X/Z basis.
    nXZplus : float, array
        Upper bound on the expected number of events per intensity in the
        sifted X/Z basis.

    """
    term2   = np.sqrt(0.5*nXZ * np.math.log(21.0 / eps_s))
    nXZmin  = np.divide(np.multiply(np.exp(mu), nxz_mu - term2), P)
    nXZplus = np.divide(np.multiply(np.exp(mu), nxz_mu + term2), P)
    return nXZmin, nXZplus

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def nXZpm_inf(mu,P,nxz_mu):
    """
    Calculates the number of events in the  sifted X or Z basis, for each pulse 
    intensity in the asymptotic limit.
    Defined after Eq. (2) in [1].
        nXi[j], or nZi[j];  j = {1:3}

    Parameters
    ----------
    mu : float, array
        Pulse intensities.
    P : float, array
        Probability of Alice preparing a pulse intensity.
    nxz_mu : float, array
        Number of events per intensity for the sifted X/Z basis.

    Returns
    -------
    nXZi : float, array
        The expected number of events per intensity in the
        sifted X/Z basis.

    """
    nXZi = np.divide(np.multiply(np.exp(mu), nxz_mu), P)
    return nXZi, nXZi

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def tau(n,mu,P):
    """
    Calculates the total probability that Alice prepares an n-photon state.
    Defined after Eq. (2) in [1].

    Parameters
    ----------
    n : integer
        Number of photons in the state.
    mu : float, array
        Intensities of the weak coherent pulses.
    P : float, array
        Probabilities that Alice prepares a particular intensity.

    Returns
    -------
    tau : float
        Total probability of an n-photon state.

    """
    if (not isinstance(n, int)): # Python 3.
        #if (not isinstance(n, (int, long))): # Python 2.
        print("Error! n must be an integer: ", n)
        exit(1)
    tau = 0
    for jj in range(len(mu)):
        tau += np.math.exp(-mu[jj]) * mu[jj]**n * P[jj]
    tau = tau / np.math.factorial(n)
    return tau

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def s0(mu,P,nMin):
    """
    Calculates the approximate number of vacuum events in the sifted X or Z 
    basis.
    See Eq. (2) in [1].

    Parameters
    ----------
    mu : float, array
        Pulse intensities.
    P : float, array
        Probabilities that Alice prepares a particular intensity.
    nMin : float, array
        Lower bound on the expected number of events per intensity in the
        sifted X/Z basis.

    Returns
    -------
    float
        The number of vacuum events in the sifted X or Z basis.

    """
    return tau(0,mu,P) * (mu[1]*nMin[2] - mu[2]*nMin[1]) / (mu[1] - mu[2])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def s1(mu,P,nMin,nPlus,s0=None):
    """
    Calculates the number of single photon events in the sifted X or Z basis.
    See Eq. (3) in [1].

    Parameters
    ----------
    mu : float, array
        Pulse intensities.
    P : float, array
        Probabilities that Alice prepares a particular intensity.
    nMin : float, array
        Lower bound on the expected number of events per intensity in the
        sifted X/Z basis.
    nPlus : float, array
        Upper bound on the expected number of events per intensity in the
        sifted X/Z basis.
    s0 : float, optional
        The number of vacuum events in the sifted X or Z basis. 
        The default is None.

    Returns
    -------
    float
        The number of single photon events in the sifted X or Z basis.

    """
    if (s0):
        # Use the value of s0 provided
        return tau(1,mu,P)*mu[0]* ( nMin[1] - nPlus[2] - \
                                   (mu[1]**2 - mu[2]**2) / mu[0]**2 * \
                                   (nPlus[0] - s0 / tau(0,mu,P)) ) / \
                            (mu[0] * (mu[1] - mu[2]) - mu[1]**2 + mu[2]**2)
    else:
        # Calculate s0
        return tau(1,mu,P)*mu[0]* ( nMin[1] - nPlus[2] - \
                                   (mu[1]**2 - mu[2]**2) / mu[0]**2 * \
                                   (nPlus[0] - s0(mu,P,nMin) / tau(0,mu,P)) ) / \
                            (mu[0] * (mu[1] - mu[2]) - mu[1]**2 + mu[2]**2)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def mXZpm(mu,P,mXZj,eps_s):
    """
    Calculates the upper and lower bounds on the number of errors in the 
    sifted X or Z basis, for each pulse intensity, using the Chernoff bounds.
    Defined after Eq. (4) in [1].
        mXplus[j] and mXmin[j], or mZplus[j] and mZmin[j];  j = {1:3}

    Parameters
    ----------
    mu : float, array
        Pulse intensities.
    P : float, array
        Probability of Alice preparing a pulse intensity.
    mXZj : float, array
        Number of errors per intensity for the sifted X/Z basis.
    eps_s : float
        The secrecy error; the key is eps_s secret.

    Returns
    -------
    mXZmin : float, array
        Lower bound on the expected number of errors per intensity in the
        sifted X/Z basis.
    mXZplus : float, array
        Upper bound on the expected number of errors per intensity in the
        sifted X/Z basis.

    """
    log_21es = np.math.log(21.0 / eps_s)
    term_m  = 0.5*log_21es + np.sqrt(2*mXZj*log_21es + 0.25*log_21es**2)
    term_p  = log_21es + np.sqrt(2*mXZj*log_21es + log_21es**2)

    mXZmin  = np.divide(np.multiply(np.exp(mu), mXZj - term_m), P)
    mXZplus = np.divide(np.multiply(np.exp(mu), mXZj + term_p), P)
    return mXZmin, mXZplus

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def mXZpm_HB(mu,P,mXZj,mXZtot,eps_s):
    """
    Calculates the upper and lower bounds on the number of errors in the 
    sifted X or Z basis, for each pulse intensity, using the Hoeffding bound.
    Defined after Eq. (4) in [1].
        mXplus[j] and mXmin[j], or mZplus[j] and mZmin[j];  j = {1:3}

    Parameters
    ----------
    mu : float, array
        Pulse intensities.
    P : float, array
        Probability of Alice preparing a pulse intensity.
    mXZj : float, array
        Number of errors per intensity for the sifted X/Z basis.
    mXZtot : float, array
        Total number of errors in the sifted X/Z basis.
    eps_s : float
        The secrecy error; the key is eps_s secret.

    Returns
    -------
    mXZmin : float, array
        Lower bound on the expected number of errors per intensity in the
        sifted X/Z basis.
    mXZplus : float, array
        Upper bound on the expected number of errors per intensity in the
        sifted X/Z basis.

    """
    term2   = np.sqrt(0.5*mXZtot * np.log(21.0 / eps_s))
    mXZmin  = np.divide(np.multiply(np.exp(mu), mXZj - term2), P)
    mXZplus = np.divide(np.multiply(np.exp(mu), mXZj + term2), P)
    return mXZmin, mXZplus

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def mXZpm_inf(mu,P,mXZj):
    """
    Calculates the Number of errors in the sifted X or Z basis, for each pulse 
    intensity, in the asymptotic limit.
    Based on Eq. (4) in [1].
        mX[j], or mZ[j];  j = {1:3}

    Parameters
    ----------
    mu : float, array
        Pulse intensities.
    P : float, array
        Probability of Alice preparing a pulse intensity.
    mXZj : float, array
        Number of errors per intensity for the sifted X/Z basis.

    Returns
    -------
    mXZi : float, array
        Expected number of errors per intensity in the sifted X/Z basis.

    """
    mXZi = np.divide(np.multiply(np.exp(mu), mXZj), P)
    return mXZi, mXZi

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def vxz1(mu,P,mXZmin,mXZplus):
    """
    Calculates the upper bound to the number of bit errors associated with 
    single photon events in the sifted X or Z basis.
    See Eq. (4) in [1].

    Parameters
    ----------
    mu : float, array
        Pulse intensities.
    P : float, array
        Probability of Alice preparing a pulse intensity.
    mXZmin : float, array
        Lower bound on the expected number of errors per intensity in the
        sifted X/Z basis.
    mXZplus : float, array
        Upper bound on the expected number of errors per intensity in the
        sifted X/Z basis.

    Returns
    -------
    float
        Upper bound to the number of bit errors associated with single photon 
        events in the sifted X/Z basis.

    """
    return tau(1,mu,P)*(mXZplus[1] - mXZmin[2]) / (mu[1] - mu[2])
        
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def mean_photon_a(P,mu):
    """
    Calculate the mean photon number for a signal sent by Alice.
    This function uses arrays.

    Parameters
    ----------
    P : float, array
        Probability Alice sends a signal intensity.
    mu : float, array
        Intensity of the pulses.

    Returns
    -------
    float
        Mean siganl photon number.

    """
    return np.dot(P, mu)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def mean_photon_v(pk1,pk2,pk3,mu1,mu2,mu3):
    """
    Calculate the mean photon number for a signal sent by Alice.
    This function uses individual values.

    Parameters
    ----------
    pk1 : float
        Probability that Alice prepares a signal with intensity 1.
    pk2 : float
        Probability that Alice prepares a signal with intensity 2.
    pk3 : float
        Probability that Alice prepares a signal with intensity 3.
    mu1 : float
        Intensity 1.
    mu2 : float
        Intensity 2.
    mu3 : float
        Intensity 3.

    Returns
    -------
    float
        Mean signal photon number.

    """
    return pk1*mu1 + pk2*mu2 + pk3*mu3