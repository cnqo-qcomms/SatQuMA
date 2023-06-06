# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:45:04 2023

@author: ksb20199
"""
import numpy as np



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
    return PAxz*PBxz*Npulse*P_times_Dj

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def mXZ(PAxz,PBxz,Npulse,P_times_ej):
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
    return PAxz*PBxz*Npulse*P_times_ej
    # return np.divide(np.multiply(P_dot_ej, np.sum(nxz, axis=0)), P_dot_Dj)

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
    term_m  = 0.5*log_21es + np.sqrt(2*nxz_mu*log_21es + 0.25*log_21es**2)
    term_p  = log_21es + np.sqrt(2*nxz_mu*log_21es + log_21es**2)

    nXZmin  = np.divide(np.multiply(np.exp(mu), nxz_mu - term_m), P)
    nXZplus = np.divide(np.multiply(np.exp(mu), nxz_mu + term_p), P)
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

from scipy.stats import binom
def FInv(nX, QBERx, eps_c):
    """
    Calculates the quantile function, or inverse cumulative distribution
    function for a binomial distribution, nCk p^k (1 - p)^(n-k), where
    n = floor(nX), p = 1-QBERx, k \propto eps_c

    Parameters
    ----------
    nX : float/integer
        Number of events in sifted X basis.
    QBERx : float
        Quantum bit error rate in sifted X basis.
    eps_c : float
        Correctness error in the secure key.

    Returns
    -------
    float
        Quartile function of binomial distribution.

    """
    return binom.ppf(eps_c * (1. + 1.0 / np.sqrt(nX)), int(nX), 1.0 - QBERx)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def h(x):
    """
    Evaluates the binary entropy function.
    Defined after Eq. (1) in [1].

    Parameters
    ----------
    x : float
        Function argument.

    Returns
    -------
    h : float
        Binary entropy.

    """
    h = -x*np.math.log(x, 2) - (1 - x)*np.math.log(1 - x, 2)
    return h

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def logM(nX, QBERx, eps_c):
    """
    This function os used to approximate the amount of secure key that is
    required to perform the error correction in bits.
    See Eq. (6) in [3].

    Parameters
    ----------
    nX : float
        The total number of measured events in the sifted X basis.
    QBERx : float
        The quantum bit error rate in the sifted X basis.
    eps_c : float
        The correctness error.

    Returns
    -------
    lM : float
        Number of bits for error correction.

    """
    lM = nX * h(QBERx) + (nX * (1.0 - QBERx) - FInv(int(nX), QBERx, eps_c) - \
                          1) * np.math.log((1.0 - QBERx) / QBERx) - \
        0.5*np.math.log(nX) - np.math.log(1.0 / eps_c)
    return lM

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def ErrorCorrect(val,fEC):
    """
    Calculates the error correction parameter \lambda_{EC}. Typical val is 1.16.
    Defined in Sec. IV of [1].

    Parameters
    ----------
    val : float
        Error correction factor.
    fEC : float
        Error correction efficiency.

    Returns
    -------
    float
        Error correction parameter.

    """
    return val * fEC

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def gamma(a,b,c,d):
    """
    Correction term. More info?
    Defined after Eq. (5) in [1].

    Parameters
    ----------
    a : float
        Argument 1.
    b : float
        Argument 2.
    c : float
        Argument 3.
    d : float
        Argument 4.

    Returns
    -------
    g : float
        Output value.

    """
    g1 = max((c + d) * (1 - b) * b / (c*d * np.math.log(2)), 0.0)
    g2 = max((c + d) * 21**2 / (c*d * (1 - b) * b*a**2), 1.0)
    g  = np.math.sqrt(g1 * np.math.log(g2, 2))
    return g

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def heaviside(x):
    """
    Heaviside step function: x -> x'\in{0,1} 

    Parameters
    ----------
    x : float
        Argument, to be corrected using the step function.

    Returns
    -------
    integer
        Binary step output.

    """
    if (x < 0):
        return 0
    else:
        if (x > 0):
            return 1
        
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



boundFunc = 'Hoeffding'
errcorrFunc = 'Block'
mu3 = 0.0 # intensity 3
ls = 0.0 # excess loss
eta_d = 0.8 # detector efficiency
Pec = 1e-8 # detector dark counts + background counts
Pap = 0.001 # afterpulse probability
QBERI = 0.003 # intrisic quantum bit error rate (QBER)
Npulse = 1e8 # no. of sent pulses
eps_s = 1e-15 # secrecy parameter
eps_c = 1e-9 # correctness parameter
num_zero = 1e-10 # numerical value to use when denominators are zero
from sys import float_info
# Extract the smallest float that the current system can:
num_min = float_info.epsilon # round (relative error due to rounding)
NoPass = 1 # number of overpasses


def key_length(x, eta_eff):
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
    # Excess loss
    etaExcess = 10**(-ls / 10.0)
    # Total efficiency = excess loss + detector efficiency + channel efficiency
    eta = etaExcess*eta_d*eta_eff

    # Exponential loss decay function
    exp_loss_jt = np.exp(-mu*eta)

    ###########################################################################

    # Probability of having a detection event per intensity [mu1,mu2,mu3]
    Dj = DRate_j(eta,Pap,Pec,exp_loss_jt)

    # Probability of having a bit error per intensity [mu1,mu2,mu3]
    ej = error_j(Dj,Pap,Pec,QBERI,exp_loss_jt)
    ###########################################################################
    ## Store some useful array products
    ###########################################################################
    # Product of each weak coherent pulse probability and the error rate
    P_times_ej   = np.multiply(P, ej)

    # product of each weak coherent pulse probability and the detection rate
    P_times_Dj = np.transpose(np.multiply(np.transpose(Dj), P))

    ###########################################################################
    ## Estimate count statistics
    ###########################################################################
    # Number of events in the sifted X basis for each intensity [mu1,mu2,mu3]
    nx_mu = nxz(Px, Px, Npulse, P_times_Dj)
    # Number of events in the sifted Z basis for each intensity [mu1,mu2,mu3]
    nz_mu = nxz(1 - Px, 1 - Px, Npulse, P_times_Dj)

    # Total number of events in the sifted X basis
    nX = np.sum(nx_mu)
    # Total number of events in the sifted Z basis
    nZ = np.sum(nz_mu) # Not used but written out
    
    # Number of errors in the sifted X basis for each intensity [mu1,mu2,mu3]
    mx_mu = mXZ(Px, Px, Npulse, P_times_ej)
    # Number of errors in the sifted Z basis for each intensity [mu1,mu2,mu3]
    mz_mu = mXZ(1 - Px, 1 - Px, Npulse, P_times_ej)
    # Total number of errors in the sifted X basis
    mXtot = np.sum(mx_mu)
    mZtot = np.sum(mz_mu)
    
    
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
        mZmin, mZplus = mXZpm(mu,P,mz_mu,eps_s)
    elif boundFunc in ['Hoeffding','hoeffding']:
        # Use Hoeffding bound
        # Returns the (3,) arrays nXmin, nXplus
        nXmin, nXplus = nXZpm_HB(mu,P,nx_mu,nX,eps_s)
        # Returns the (3,) arrays nZmin, nZplus
        nZmin, nZplus = nXZpm_HB(mu,P,nz_mu,nZ,eps_s)
        # Returns the (3,) arrays mXmin and mXplus
        #mXmin, mXplus = mXZpm_HB(mu,P,mXj,mXtot,eps_s) # Not used
        # Returns the (3,) arrays mZmin and mZplus
        mZmin, mZplus = mXZpm_HB(mu,P,mz_mu,mZtot,eps_s)
    elif boundFunc in ['Asymptotic','asymptotic']:
        # Use asymptotic bounds - no bounds
        # Returns the (3,) arrays nXmin, nXplus
        nXmin, nXplus = nXZpm_inf(mu,P,nx_mu)
        # Returns the (3,) arrays nZmin, nZplus
        nZmin, nZplus = nXZpm_inf(mu,P,nz_mu)
        # Returns the (3,) arrays mXmin and mXplus
        #mXmin, mXplus = mXZpm_inf(mu,P,mXj,eps_s) # Not used
        # Returns the (3,) arrays mZmin and mZplus
        mZmin, mZplus = mXZpm_inf(mu,P,mz_mu)
    else:
        # Use Chernoff bounds
        # Returns the (3,) arrays nXmin, nXplus
        nXmin, nXplus = nXZpm(mu,P,nx_mu,eps_s)
        # Returns the (3,) arrays nZmin, nZplus
        nZmin, nZplus = nXZpm(mu,P,nz_mu,eps_s)
        # Returns the (3,) arrays mXmin and mXplus
        #mXmin, mXplus = mXZpm(mu,P,mXj,eps_s) # Not used
        # Returns the (3,) arrays mZmin and mZplus
        mZmin, mZplus = mXZpm(mu,P,mz_mu,eps_s)

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
        lambdaEC = logM(nX, QBERx, eps_c)
    elif errcorrFunc in ['Block','block']:
        lambdaEC = 1.22 * nX * h(QBERx)
    elif errcorrFunc in ['mXtot','mxtot']:
        lambdaEC = 1.22 * mXtot
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
        phi_x = min(ratio, 0.5)
        # Secret key length in the asymptotic regime
        l = max((sx0 + sx1 * (1 - h(phi_x)) - lambdaEC) * 
             heaviside(mu[0] - mu[1] - mu[2]) * heaviside(P[2]), 0.0)
    else:
        # Phase error rate for single photon events in the sifted X basis=
        phi_x = min(ratio + gamma(eps_s,ratio,sz1,sx1), 0.5)
        # Secret key length in the finite regime
        l = max((sx0 + sx1 * (1 - h(phi_x)) - lambdaEC -
             6*np.math.log(21.0 / eps_s, 2) - np.math.log(2.0 / eps_c, 2)) * 
             heaviside(mu[0] - mu[1] - mu[2]) * heaviside(P[2]), 0.0)
        l = l / NoPass # Normalise by the number of satellite passes used=
    print(l)
    return l, Dj, ej, nx_mu, nz_mu, mx_mu, mz_mu, QBERx, phi_x, lambdaEC, sx0, sx1, vz1, phi_x


x = [0.7611, 0.7501, 0.1749, 0.7921, 0.1707] # Px, p_mu1, p_mu2, mu1, mu2
eta_eff = 0.02 # channel efficiency

SKL, _, _, _, _, _, _, _, _, _, _, _, _, _ = key_length(x, eta_eff)

print('SKL =', int(SKL), 'bits')