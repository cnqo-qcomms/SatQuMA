# -*- coding: utf-8 -*-
"""
This script calculates the secure key length for an aysmmetric BB84 security
protocol with weak coherent pulses and three signal states (or 2 'decoy' 
states).
The 5 main protocol parameters can be either optimised or specified:
    Px  = X basis polarisation bias (Alice & Bob)
    pk1 = Probability Alice sends a signal with intensity 1
    pk2 = Probability Alice sends a signal with intensity 2
    mu1 = Intensity of signal 1
    mu2 = Intesnity of signal 2

This version employs either the Chernoff or Hoeffding bound.

This script is based on the Mathematica script SatQKD_finite_key.nb written by
J. S. Sidhu and T. Brougham.
Both are primarily based upon the paper,

[1] C. C. W. Lim, M. Curty, N. Walenta, F. Xu, and H. Zbinden, "Concise 
security bounds for practical decoy-state quantum key distribution", Phys. Rev.
A, 89, 022307 (2014),

with the bounds on the statistical fluctuations for the number of n-photon 
events taken from,
 
[2] H.-L. Yin, M.-G. Zhou, J. Gu, Y.-M. Xie, Y.-S. Lu, and Z.-B. Chen, "Tight 
security bounds for decoy-state quantum key distribution", Sci. Rep., vol. 10, 
14312 (2020),

and the estimation of the error correction term from,

[3] M. Tomamichel, J. Martinez-Mateo, C. Pacher, and D. Elkouss, "Fundamental 
finite key limits for one-way information reconciliation in quantum key 
distribution," Quant. Inf. Proc., vol. 16, 280, (2017).

                                ----------

When running the code the user should check the values in the input sections 
marked (1) and (2) below, where:
    (1) The type of calculation is selected and the optimised/specified 
        parameters are initialised.
    (2) The global parameters defining the system are set.
    
"""
import numpy as np
from sys import exit
from time import perf_counter, process_time

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#******************************************************************************
#******************************************************************************
#                      ---  USER INPUT SECTION (1)  ---
#     Select the type of secure key length calculation and intialise the
#     parameters
#******************************************************************************
#******************************************************************************

F_or_T = [False, True] # List used to switch between False or True values

#******************************************************************************
# Select SKL calculation type via tOptimise
#******************************************************************************
#    True:  Optimise over the main protocol parameters.
#    False: Specify the main protocol parameters.
#******************************************************************************
tOptimise = F_or_T[1]  # False (0) or True (1)

if (tOptimise):
    #**************************************************************************
    # Calculate the secure key length by optimising over the protocol
    # parameters.
    #**************************************************************************
    
    #**************************************************************************
    # Limit the bounds for each parameter for the optimisation search.
    # Each row of the array xb gives the lower and upper bound [lb, ub] for 
    # each of the parameters in the order [Px,pk1,pk2,mu1,mu2]
    #**************************************************************************
    xb = np.array([[0.3,1.0],[0.6,0.9999],[0.0,0.4],[0.3,1.0],[0.1,0.5]])
    #xb = np.array([[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0]]) # Default
    
    #**************************************************************************
    # Select parameter intialisation procedure via tInit
    #**************************************************************************
    #    True:  Provide initial values for optimised parameters.
    #    False: Randomly select initial values for optimised parameters
    #           based on the protocol bounds and constraints.
    #**************************************************************************
    tInit = F_or_T[1]  # False (0) or True (1)

    if (tInit):
        #**********************************************************************
        # Provide initial values for optimised parameters
        #**********************************************************************
        Px_i  = 0.5  # Asymmetric polarisation probability
        pk1_i = 0.7  # Probability Alice prepares intensity 1
        pk2_i = 0.1  # Probability Alice prepares intensity 2
        mu1_i = 0.8  # Intensity 1
        mu2_i = 0.3  # Intensity 2

else:
    #**************************************************************************
    # Calculate the secure key length using specified values for the protocol 
    # parameters
    #**************************************************************************
    ### For ls = 13, dt = 221, EC = mXtot, Chernoff
    Px_i  = 0.7611  # Asymmetric polarisation probability
    pk1_i = 0.7501  # Probability Alice prepares intensity 1
    pk2_i = 0.1749  # Probability Alice prepares intensity 2
    mu1_i = 0.7921  # Intensity 1
    mu2_i = 0.1707  # Intensity 2

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#******************************************************************************
#******************************************************************************
#                      ---  USER INPUT SECTION (2)  ---
#     Define the global parameters used to calculate the secure key length
#******************************************************************************
#******************************************************************************

#******************************************************************************
# Input file options
#******************************************************************************
# File containing loss data (for given xi value below)
loss_file = 'FS_loss_XI0.csv'
# Path to loss file
loss_path = ''
lc        = 3 # Column containing loss data in file (counting from 1)

#******************************************************************************
# Output file options
#******************************************************************************
# Write output to CSV file?
tWriteData = F_or_T[1]  # False (0) or True (1)
outpath    = ''    # Path for output file (empty = current directory)
outfile    = 'out' # Name for output file (minus .csv)
# Write out a sorted data set?
tSortData  = F_or_T[1]  # False (0) or True (1)
# Write out only max values of SKL for dt?
tOptiData  = F_or_T[1] # False (0) or True (1)
# Print values to StdOut during calculations?
tPrint     = F_or_T[1]  # False (0) or True (1)

#******************************************************************************
# Define (inclusive) range for looped parameters: dt and ls
#******************************************************************************
# Index for windowing time slot arrays, e.g. A(t)[t0-dt:t0+dt], with dt <= 346
#dt_range = np.array([0, 346, 10]) # Start, stop, step index (defualt)
dt_range = np.array([211, 221, 10]) # Start, stop, step
# Set a minimum elevation angle for transmission, this value will override
# the values corresponding to dt_range[1] if necessary.
min_elev = 10.0 # Minimum elevation transmission angle (degs)

# Excess system loss in dB
#ls_range = np.array([0, 13, 4])   # Start, stop, step value
ls_range = np.array([12, 13, 1])   # Start, stop, step 

#******************************************************************************
# Fixed system parameters
#******************************************************************************
xi  = 0.0 # Angle between OGS zenith and satellite (from Earth's centre) [rad.]
#mu3 = 10**(-9) # Weak coherent pulse 3 intensity, mu_3
mu3 = 0
# Prescribed errors in protocol correctness and secrecy
eps_c = 10**(-15)
eps_s = 10**(-9)
# Afterpulse probability
Pap = 0.001
# Polarization error (QBER_I)
PolError = 0.005
# Dark count probability
Pdc = 5*10**(-7)
# Number of satellite passes
NoPass = 1
# Repetition rate of the source in Hz
Rrate  = 1*10**(8) # Should match with value used to generate loss file
# Number of pulses sent in Hz
Npulse = NoPass*Rrate

#******************************************************************************
# NOTE: All nominal losses have been moved into FS_loss_XI<xi>.csv for now and 
# so we simply set the efficiencies listed below to unity.
#******************************************************************************
# Detector efficiency
eta_d = 1.0 # Typically 40-60%
# Internal telescope transmitter loss
eta_transmitter = 1.0 # Typically ~50%
# Combined losses
eta = eta_d * eta_transmitter

#******************************************************************************
# Advanced/additional parameters
#******************************************************************************
# Use the Chernoff bounds when estimating statistical fluctuations in the
# count statistics? Otherwise use the Hoeffding bound.
tChernoff = F_or_T[1]  # False (0) or True (1)

# Select the method for estimating the number of bits sacrificed for error
# correction, listed below in order of decreasing precision (increasing
# smoothness).
#   'logM'  = logM(nX, QBERx, eps_c) = errcorrOpts[0]
#   'block' = 1.16 * nX * h(QBERx)   = errcorrOpts[1]
#   'mXtot' = 1.16 * mXtot           = errcorrOpts[2]
#   'None'  = 0                      = errcorrOpts[3]
errcorrOpts = ['logM','block','mXtot','None']
errcorrFunc = errcorrOpts[2] # Select a method from the list above
# Compare SKL optimised with/without EC (for optimised calculation only). 
tCompareEC  = F_or_T[1]  # False (0) or True (1)

# Numerical value to use when denominator values are potentially zero
num_zero = 10**(-10) # Note to self: Use num_min below?

# Optimiser options: minimize, method='trust-constr'
xtol      = 1e-8   # Tolerance to terminate by change of independent variable(s)
gtol      = 1e-10  # Tolerance to terminate Lagrangian gradient
btol      = 1e-8   # Threshold on the barrier parameter for termination
Nmax      = 1000   # Max No. of iterations
const_pen = 1.0    # Initial constraint penalty parameter (default = 1.0)
tr_rad    = 1.0    # Initial trust radius (default = 1.0)
barr_par  = 0.1    # Initial barrier parameter (default = 0.1)
barr_tol  = 0.1    # Initial barrier tolerance for barrier sub- (default = 0.1)

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#******************************************************************************
#******************************************************************************
#                   ---  END OF USER INPUT SECTIONS  ---
# ANY EDITS MADE BEYOND THIS POINT MAY AFFECT THE OPERATION OF THE SOFTWARE
#******************************************************************************
#******************************************************************************

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#******************************************************************************
#******************************************************************************
# Read in free-space loss data from CSV file
#******************************************************************************
#******************************************************************************
# Read from a local file and skip first line and take data only from the 
# specified column. The free space loss should be arranged in terms of time
# slots, t.
cvs = np.loadtxt(loss_path+loss_file, delimiter= ',', skiprows=1, 
                 usecols=(0, 1, lc-1,))

# Find the time slot at the centre of the pass where t = 0.
time0pos = np.where(cvs[:,0] == 0)[0][0]

# Maximum elevation angle (degs) of satellite pass
max_elev = np.degrees(cvs[time0pos,1])

# Free space loss in dB (to be converted to efficiency)
# Returns the (t,) array FSeff, where t is the total number of time-slots
FSeff = cvs[:,2]

# Nominal system loss: based on zenith coupling efficiency and nominal losses
sysLoss = -10*(np.math.log10(FSeff[time0pos]) + np.math.log10(eta))

# Check that minimum elevation angle specified is valid
if (min_elev < 0.0 or min_elev > 90):
    print('Error! Minimum elevation angle out of bounds:',min_elev)
    print('Angle should be between 0 and 90 degrees.')
    exit(1)
# Find first dt value corresponding to an elevation greater than min_elev
minElevpos = np.where(cvs[:,1] >= np.radians(min_elev))[0][0] # Index of first value
dt_elev    = cvs[minElevpos,0] # Max value of dt less than, or equal to, the
                               # minimum elevation angle

# Check dt_range is within bounds
dt_max = int(0.5*(len(FSeff)-1))       # Maximum time window half-width
dt_range[dt_range < 0] = 0             # All values must be positive
dt_range[dt_range > dt_max] = dt_max   # Max cannot exceed No. of time-slots
dt_range[dt_range > dt_elev] = dt_elev # Limit range by minimum elevation value

# Get minimum elevation for transmission (possibly greater than value specified)
minElevpos = np.where(cvs[:,0] <= dt_range[1])[0][0] # Index of first value
min_elev   = np.degrees(cvs[minElevpos,1]) # Minimum elevation (degs)

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#******************************************************************************
#******************************************************************************
# Set some machine dependent constants
#******************************************************************************
#******************************************************************************
from sys import float_info
# Extract the smallest float that the current system can:
num_min = float_info.epsilon # round (relative error due to rounding)
#num_min = float_info.min     # represent
# Extract the largest float that the current system can represent
num_max = float_info.max

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#******************************************************************************
#******************************************************************************
# Sub-functions used to determine the secure key length and to process data
#******************************************************************************
#******************************************************************************

def check_constraints(Px,pk1,pk2,mu1,mu2,mu3):
    """
    Check that the parameters are within the bounds and constraints of the
    asymmetric BB84 protocol with weak coherent pulses with 2 'decoy' states.
    Stops the script if any bounds or constraints are violated.

    Parameters
    ----------
    Px : float
        Asymmetric polarisation probability.
    pk1 : float
        Probability Alice sends pulse intensity 1.
    pk2 : float
        Probability Alice sends pulse intensity 2.
    mu1 : float
        Intensity of pulse 1.
    mu2 : float
        Intensity of pulse 2.
    mu3 : float
        Intensity of pulse 3.

    Returns
    -------
    None.

    """
    # Constraint 1: Check polarisation basis probabilities are valid.
    if (Px >= 1.0 or Px <= 0.0):
        print("Error! Constraint 1 < Px < 0: ", Px)
        exit(1)
    # Constraint 2: Check probability of pulse with intensity 1 is in bounds.
    if (pk1 >= 1.0 or pk1 <= 0.0):
        print("Error! Constraint 1 < pk1 < 0: ", pk1)
        exit(1)
    # Constraint 3: Check probability of pulse with intensity 2 is in bounds.
    if (pk2 >= 1.0 or pk2 <= 0.0):
        print("Error! Constraint 1 < pk2 < 0: ", pk2)
        exit(1)
    # Constraint 4: Check sum of probabilities for intensity 1 & 2 are less
    # than unity.
    if ((pk1 + pk2) >= 1.0):
        print("Error! Constraint (pk1 + pk2) < 1: ", pk1 + pk2)
        exit(1)
    # Constraint 5: Check value of intensity 1 is in bounds.
    if (mu1 >= 1.0 or mu1 <= 0.0):
        print("Error! Constraint 1 < mu1 < 0: ", mu1)
        exit(1)
    # Constraint 6: Check value of intensity 2 is in bounds.
    if (mu2 >= 1.0 or mu2 <= 0.0):
        print("Error! Constraint 1 < mu2 < 0: ", mu2)
        exit(1)
    # Constraint 7: Check values of all intensities are in bounds.
    if ((mu1 - mu3) <= mu2):
        print("Error! Constraint (mu1-mu3) > mu2: ", (mu1-mu3), mu2)
        exit(1)
    # Constraint 8: Check values of intensities 2 & 3 are in bounds.
    if (mu2 <= mu3):
        print("Error! Constraint mu2 > mu3: ", mu2, mu3)
        exit(1)
    return None

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def bool_constraints(Px,pk1,pk2,mu1,mu2,mu3):
    """
    Check that the parameters are within the bounds and constraints of the
    asymmetric BB84 protocol with weak coherent pulses with 2 'decoy' states.
    Returns a boolean array corresponding to each of the constraints.

    Parameters
    ----------
    Px : float
        Asymmetric polarisation probability.
    pk1 : float
        Probability Alice sends pulse intensity 1.
    pk2 : float
        Probability Alice sends pulse intensity 2.
    mu1 : float
        Intensity of pulse 1.
    mu2 : float
        Intensity of pulse 2.
    mu3 : float
        Intensity of pulse 3.

    Returns
    -------
    C : boolean, array-like.
        Do the parameters satisfy the constraints? True or False

    """
    C = np.array([1,1,1,1,1,1,1,1], dtype=bool) # Initialise array as True
    
    # Constraint 1: Check polarisation basis probabilities are valid.
    if (Px >= 1.0 or Px <= 0.0):
        C[0] = False
    # Constraint 2: Check probability of pulse with intensity 1 is in bounds.
    if (pk1 >= 1.0 or pk1 <= 0.0):
        C[1] = False
    # Constraint 3: Check probability of pulse with intensity 2 is in bounds.
    if (pk2 >= 1.0 or pk2 <= 0.0):
        C[2] = False
    # Constraint 4: Check sum of probabilities for intensity 1 & 2 are less
    # than unity.
    if ((pk1 + pk2) >= 1.0):
        C[3] = False
    # Constraint 5: Check value of intensity 1 is in bounds.
    if (mu1 >= 1.0 or mu1 <= 0.0):
        C[4] = False
    # Constraint 6: Check value of intensity 2 is in bounds.
    if (mu2 >= 1.0 or mu2 <= 0.0):
        C[5] = False
    # Constraint 7: Check values of all intensities are in bounds.
    if ((mu1 - mu3) <= mu2):
        C[6] = False
    # Constraint 8: Check values of intensities 2 & 3 are in bounds.
    if (mu2 <= mu3):
        C[7] = False
    return C

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def DRate_j(eta,Pap,Pdc,exp_loss_jt):
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
    Pdc : float
        Dark count probability.
    exp_loss_jt : float, array
        Loss, per intensity per time slot, decay function.

    Returns
    -------
    float, array
        Expected detection rate.

    """
    return (1 + Pap)*(1 - (1 - 2*Pdc)*exp_loss_jt)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def error_j(Dj,Pap,Pdc,PolError,exp_loss_jt):
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
    Pdc : float
        Dark count probability.
    PolError : float
        Errors in polarisation basis.
    exp_loss_jt : float, array
        Loss, per intensity per time slot, decay function.

    Returns
    -------
    float, array
        Error rate per intensity per time slot.

    """
    return Pdc + (0.5*Pap*Dj) + PolError*(1 - exp_loss_jt)

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

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def writeDataCSV(data,outpath,outfile,out_head=None):
    """
    Write out data to a CSV file

    Parameters
    ----------
    data : float, array-like
        Data array containing parameters, SKL, and protocol metrics.
    outpath : string
        Path for output file.
    outfile : string
        Name for output file.
    out_head : string, optional
        Header for data file

    Returns
    -------
    None.

    """
    if (out_head is not None):
        #nhead = out_head.count(',') + 1
        nhead = len(out_head.split(',')) # Split header at every comma
        if (data.shape[1] != nhead):
            print('Warning: No. of fields does not match number of headings in', 
                  'output file:',outfile+'.csv')
            print('No. fields =',data.shape[1],', No. headings =',nhead)
    if (outpath == ''):
        filename = outfile + '.csv'
    else:
        filename = outpath + '/' + outfile + '.csv'
    print('Saving data as:',filename)
    np.savetxt(filename,data,delimiter=',',header=out_head) 
    return None

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def sort_data(data,header,sort_tags,rev_sort,sortkind='mergesort'):
    """
    Sort a data array according to a list of data tags which are taken from a
    header string. The array is sorted in the order that the tags are listed.

    Parameters
    ----------
    data : float, array-like
        Each column of the array contains a different variable.
    header : string
        Comma separated header string identifying the variables in each column
        of the data array.
    sort_tags : string, list
        List of strings defining the data columns to sort by.
    rev_sort : logical,list
        Perform a reverse sort? Should have the same length as sort_tags.
    sortkind : string, optional
        The type of sort to perform. The default is 'mergesort'.

    Returns
    -------
    sortdata : float, array-like
        The sorted data array.

    """
    tags     = header.split(',') # Split the header string into separate elements of a list
    nsort    = len(sort_tags)    # Number of columns to sort by
    for ii in range(0,nsort,1):
        try:
            sc = tags.index(sort_tags[ii]) # Check tags match headings
        except ValueError:
            print('Error! Sort tag not recognised:',sort_tags[0])
            return None
    if (len(sort_tags) != len(rev_sort)):
        print("Error! Lists 'sort_tags' and 'rev_sort' have different lengths")
        print('len(sort_tags) =',len(sort_tags),' len(rev_sort) =',len(rev_sort))
        return None
    sc       = tags.index(sort_tags[0]) # First column to sort by
    sortdata = data # Take a copy to change it
    if rev_sort[0]:
        # Reverse sort data array
        sortdata = sortdata[sortdata[:,sc].argsort()[::-1]] # sort by column scol
    else:
        # Sort data array
        sortdata = sortdata[sortdata[:,sc].argsort()] # sort by column scol
    ii = 1
    for s_tag in sort_tags[1:]:
        sc = tags.index(s_tag) # Next column to sort by
        if rev_sort[ii]:
            sortdata = sortdata[sortdata[:,sc].argsort(kind=sortkind)[::-1]]
        else:
            sortdata = sortdata[sortdata[:,sc].argsort(kind=sortkind)]
        ii += 1
    return sortdata

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#******************************************************************************
#******************************************************************************
# Function(s) to calculate the secure key length
#******************************************************************************
#******************************************************************************

# Secure Key Length (SKL) function - for direct calculation
def key_length(x, *args):
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
    mu = (x[3],x[4],mu3)
    # Probability of Alice sending a pulse of intensity mu_k
    P  = (x[1],x[2],1 - x[1] - x[2])

    ###########################################################################
    ## Define total system loss and convert to efficiency
    ###########################################################################
    # Excess loss in decibels
    etaExcess = 10**(-ls / 10.0)
    # Scale excess loss by detector efficiency
    eta = etaExcess*eta_d
    # Take the outer product of FSeff, the free space loss, and mu, the pulse
    # intensities, where FSeff has a length t.
    # Use dt to take a window of t values from FSeff.
    # Returns the (3,t) array mu_FSeff
    mu_FSeff = np.outer(mu, FSeff[time0pos-dt:time0pos+dt+1])
    # Exponential loss decay function
    # Returns the (3,t) array exp_loss_jt
    exp_loss_jt = np.exp(-mu_FSeff*eta)

    ###########################################################################
    ## Estimate count and error rates for signals
    ###########################################################################
    # Expected detection rate (including afterpulse contributions)
    # Returns the (3,t) array Dj
    #print('mu    =',mu)
    Dj = DRate_j(eta,Pap,Pdc,exp_loss_jt)
    ###print('Dj    =',np.sum(Dj,axis=1))
    # Probability of having a bit error per intensity for a given time slot
    # Returns the (3,t) array ej
    ej = error_j(Dj,Pap,Pdc,PolError,exp_loss_jt)
    
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
    # Number of events in the sifted X basis for each time slot and
    # intensity
    # Returns the (3,t) array nx
    nx = nxz(Px, Px, Npulse, P_times_Dj)
    # Number of events in the sifted Z basis for each time slot and
    # intensity
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
    if (tChernoff):
        # Use Chernoff bounds
        # Returns the (3,) arrays nXmin, nXplus
        nXmin, nXplus = nXZpm(mu,P,nx_mu,eps_s) # Chernoff bound
        # Returns the (3,) arrays nZmin, nZplus
        nZmin, nZplus = nXZpm(mu,P,nz_mu,eps_s)
    else:   
        # Use Hoeffding bound
        # Returns the (3,) arrays nXmin, nXplus
        nXmin, nXplus = nXZpm_HB(mu,P,nx_mu,nX,eps_s) # Hoeffding bound
        # Returns the (3,) arrays nZmin, nZplus
        nZmin, nZplus = nXZpm_HB(mu,P,nz_mu,nZ,eps_s)

    # Upper and lower bounds used to estimate the number of errors in the
    # sifted X or Z basis for each intensity
    if (tChernoff):
        # Use Chernoff bounds
        # Returns the (3,) arrays mXmin and mXplus
        #mXmin, mXplus = mXZpm(mu,P,mXj,eps_s) # Not used
        # Returns the (3,) arrays mZmin and mZplus
        mZmin, mZplus = mXZpm(mu,P,mZj,eps_s) # Chernoff bound
    else:
        # Use Hoeffding bound
        # Returns the (3,) arrays mXmin and mXplus
        #mXmin, mXplus = mXZpm_HB(mu,P,mXj,mXtot,eps_s) # Not used
        # Returns the (3,) arrays mZmin and mZplus
        mZmin, mZplus = mXZpm_HB(mu,P,mZj,mZtot,eps_s) # Hoeffding bound

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
    # Phase error rate for single photon events in the sifted X basis
    phi_x = min(ratio + gamma(eps_s,ratio,sz1,sx1), 0.5)
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
        lambdaEC = 1.16 * nX * h(QBERx)
    elif errcorrFunc in ['mXtot','mxtot']:
        lambdaEC = 1.16 * mXtot
    elif errcorrFunc in ['None','none']:
        lambdaEC = 0
    else:
        lambdaEC = 0
    
    ###########################################################################
    ## Calculate the approximate length of the secret key per pass
    ## See Eq. (1) in [1].
    ###########################################################################
    l = max((sx0 + sx1 * (1 - h(phi_x)) - lambdaEC -
             6*np.math.log(21.0 / eps_s, 2) - np.math.log(2.0 / eps_c, 2)) * 
             heaviside(mu[0] - mu[1] - mu[2]) * heaviside(P[2]), 0.0)
    l = l / NoPass # Normalise by the number of satellite passes used

    return l, QBERx, phi_x, nX, nZ, mXtot, lambdaEC, sx0, sx1, vz1, sz1, mpn

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Secure Key Length (SKL) inverse function - returns (1 / SKL) - for optimizer
def key_length_inv(x, *args):
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
    # Safety check that the parameters satisfy the constraints
    C = bool_constraints(x[0],x[1],x[2],x[3],x[4],mu3)
    if (not np.all(C)):
        return num_max  # ~1/0
    
    # Calculate the secure key length (normalised by NoPass)
    l, _, _, _, _, _, _, _, _, _, _, _ = key_length(x)

    if (l > 0):
        return (1.0/l)  # Inverse key length
    elif (l == 0):
        return num_max  # ~1/0
    else:
        #return num_max  # ~1/0
        print("Error! Unexpected key length:", l)
        return l        # Negative key length, NaN?


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#******************************************************************************
#******************************************************************************
# Secure key length calculation
#******************************************************************************
#******************************************************************************

#******************************************************************************
# Initialise calculation parameters
#******************************************************************************
if (tOptimise):
    if (not tInit):
        # Randomly set the initial parameters using the parameter bounds, 
        # relations and the protocol constraints.
        Px_i  = np.random.rand() * (xb[0,1] - xb[0,0] - 2*num_min) + xb[0,0] + num_min
        pk1_i, pk2_i = 1.0, 1.0
        while (pk1_i+pk2_i >= 1.0):
            pk1_i = np.random.rand() * (xb[1,1] - xb[1,0] - 2*num_min) + xb[1,0] + num_min
            pk2_i = np.random.rand() * (min(xb[2,1],1-pk1_i) - xb[2,0] - 2*num_min) + xb[2,0] + num_min
        mu1_i = np.random.rand() * (xb[3,1] - max(xb[3,0],2*mu3) - 2*num_min) + max(xb[3,0],2*mu3) + num_min
        mu2_i = np.random.rand() * (min(xb[4,1],mu1_i) - max(xb[4,0],mu3) - 2*num_min) + max(xb[4,0],mu3) + num_min

# Check initial parameters are within the bounds and constraints of the method
check_constraints(Px_i,pk1_i,pk2_i,mu1_i,mu2_i,mu3)

# Parameters to (potentially) optimise - store initial values as array
x0 = np.array([Px_i,pk1_i,pk2_i,mu1_i,mu2_i])

#******************************************************************************
# Initialise loop ranges
#******************************************************************************
# Set the number of outputs to store
if (ls_range[0] == ls_range[1] or ls_range[2] == 0):
    nls = 1
else:
    nls = int((ls_range[1] - ls_range[0]) / ls_range[2]) + 1
if (dt_range[0] == dt_range[1] or dt_range[2] == 0):
    ndt = 1
else:
    ndt = int((dt_range[1] - dt_range[0]) / dt_range[2])+1

#******************************************************************************
# Initialise output data storage and headers
#******************************************************************************
# Header for CSV file: Data columns
header = "SysLoss,dt,SKL,QBER,phiX,nX,nZ,lambdaEC,sX0,sX1,vZ1,sZ1," + \
         "mean photon no.,polError,Pdc,Pap,NoPass,Rrate,eps_c,eps_s," + \
         "Px,P1,P2,P3,mu1,mu2,mu3,xi (deg),minElev (deg),maxElev (deg)"
# Initialise a data storage array: shape(No. of data runs, No. of metrics)   
fulldata = np.empty((nls*ndt,len(header.split(","))))

if (tOptimise):
    # Header for CSV file: Optimiser metrics
    opt_head = "1/fun,x0,x1,x2,x3,x4,opt,con_vln,grad0,grad1,grad2,grad3," + \
               "grad4,lg_gr0,lg_gr1,lg_gr2,lg_gr3,lg_gr4,Nit,nfev,njev," + \
               "nhev,Ncg,tr_rad,con_pen,barr_tol,barr_par,status,cg_stop"
    # Initialise a data storage array for optimiser metrics
    optdata = np.empty((nls*ndt,len(opt_head.split(","))))

#******************************************************************************
# Print main calculation parameters
#******************************************************************************
print('_'*60,'\n')
print('-'*60)
print('SatQuMA v1.0 alpha (prototype-monolith)')
print('-'*60)
print('Efficient BB84 security protocol')
if (tOptimise):
    print('Using optimised protocol parameters: Px, pk, mu')
    if (tInit):
        print('Protocol parameters initialised by user')
    else:
        print('Protocol parameters intialised randomly')
else:
    print('Using specified protocol parameters: Px, pk, mu')
if (tChernoff):
    print('Using Chernoff bounds for statistical fluctuations')
else:
    print('Using Hoeffding bounds for statistical fluctuations')
if errcorrFunc in ['logM','logm']:
    print('Error correction model: logM(nX, QBERx, eps_c)')
elif errcorrFunc in ['Block','block']:
    print('Error correction model: 1.16 * nX * h(QBERx)')
elif errcorrFunc in ['mXtot','mxtot']:
    print('Error correction model: 1.16 * mXtot')
elif errcorrFunc in ['None','none']:
    print('Error correction term not included')
else:
    print('Error correction term not included')
if (nls*ndt > 1):
    print('User requested',nls*ndt,'calculations:')
    print(' >',ndt,'transmission windows')
    print(' >',nls,'system loss values')
else:
    print('User requested',nls*ndt,'calculation')
print('-'*60)
print('_'*60,'\n')

#******************************************************************************
# Begin main calculations
#******************************************************************************
if (tOptimise):
    if errcorrFunc in ['None','none']:
        tCompareEC = False # Don't perform the same calculation twice
    # Sanity check on parameter bounds
    xb[xb < 0] = 0 # All bounds must be positive
    xb[xb > 1] = 1 # All bounds are less than or equal to 1
    # Calculate the secure key length by optimising over the protocol
    # parameters.
    # Build bounds arrays for x = [Px,pk1,pk2,mu1,mu2]
    lb = np.array([xb[0,0],xb[1,0],xb[2,0],xb[3,0],xb[4,0]]) + num_min # Lower bounds for parameters
    ub = np.array([xb[0,1],xb[1,1],xb[2,1],xb[3,1],xb[4,1]]) - num_min # Upper bounds for parameters
    from scipy.optimize import Bounds
    # Store the parameters bounds as an object, as required by minimize() for
    # method='trust-constr'
    bounds = Bounds(lb, ub)          # Store upper and lower bounds
    # Build linear constraint arrays for x = [Px,pk1,pk2,mu1,mu2]^T such that:
    # (lb_l)^T <= A_l x <= (ub_l)^T
    # Constraints with bounds = +/-np.inf are interpreted as one-sided
    lb_l = np.array([xb[0,0],num_min,xb[2,0],(mu3 + num_min),(mu3 + num_min)])
    ub_l = np.array([xb[0,1],(1. - num_min),xb[2,1],np.inf,np.inf])
    A_l  = np.array([[1.,0.,0.,0., 0.], \
                     [0.,1.,1.,0., 0.], \
                     [0.,0.,1.,0., 0.], \
                     [0.,0.,0.,1.,-1.], \
                     [0.,0.,0.,0., 1.]])
    # The above arrays yield the following linear constraints:
    # (1)   0   < Px  < 1      -> Assuming xb[0,:] = [0,1]
    # (2)   pk1 + pk2 < 1
    # (3)   0   < pk2 < 1      -> Assuming xb[2,:] = [0,1]
    # (4)   mu3 < mu1 - mu2
    # (5)   mu3 < mu2
    from scipy.optimize import LinearConstraint
    # Store the linear constraints as an object, as required by minimize() for
    # method='trust-constr'
    lin_constraint = LinearConstraint(A_l, lb_l, ub_l)

    # Print out the intial parameters, bounds, and algorithm options
    if tPrint:
        print('Initial protocol parameters:')
        print('Px_i =',Px_i)
        print('pk_i = (',pk1_i,pk2_i,1-pk1_i-pk2_i,')')
        print('mu_i = (',mu1_i,mu2_i,mu3,')')
        print('-'*60)
        print('Protocol parameter bounds (lower,upper):')
        print('Px   = ({0}, {1})'.format(lb[0],ub[0]))
        print('pk1  = ({0}, {1})'.format(lb[1],ub[1]))
        print('pk2  = ({0}, {1})'.format(lb[2],ub[2]))
        print('mu1  = ({0}, {1})'.format(lb[3],ub[3]))
        print('mu2  = ({0}, {1})'.format(lb[4],ub[4]))
        print('-'*60)
        print("Optimiser parameters ('trust-constr'):")
        print('Nmax =',Nmax)
        print('xtol =',xtol)
        print('gtol =',gtol)
        print('btol =',btol)
        print('Initial constraint penalty =',const_pen)
        print('Initial trust radius       =',tr_rad)
        print('Initial barrier parameter  =',barr_par)
        print('Initial barrier tolerance  =',barr_tol)
        print('-'*60)
        print('-'*60,'\n')

    from scipy.optimize import minimize
    #def hess0(x):
    #     return np.zeros((5,5))
    tc0 = perf_counter() # Start clock timer
    tp0 = process_time() # Start CPU timer
    # Perform the optimization of the secure key length
    count = 0 # Initialise calculation counter
    if tCompareEC:
        # Compare optimised SKL for, first, including error correction during
        # the optimissation procedure and then, second, for including the
        # error correction term after the optimisation procedure.
        funcEC = errcorrFunc # Store error correction function choice
        for ls in range(ls_range[0],ls_range[1]+ls_range[2],ls_range[2]):
            for dt in range(dt_range[0],dt_range[1]+dt_range[2],dt_range[2]):
                print('Calculation {0}: ls = {1}, dt = {2}'.format(count+1,ls,int(dt)))
                # Calculate optimised SKL
                res = minimize(key_length_inv, x0, method='trust-constr',
                               jac=None, hess=None, hessp=None, bounds=bounds, 
                               constraints=lin_constraint, tol=None, callback=None, 
                               options={'xtol': xtol,
                                        'gtol': gtol, 
                                        'barrier_tol': btol,
                                        'sparse_jacobian': None, 
                                        'maxiter': Nmax,
                                        'verbose': 0, 
                                        'finite_diff_rel_step': None, 
                                        'initial_constr_penalty': const_pen,
                                        'initial_tr_radius': tr_rad, 
                                        'initial_barrier_parameter': barr_par, 
                                        'initial_barrier_tolerance': barr_tol, 
                                        'factorization_method': None,
                                        'disp': False})

                if (tPrint):
                    print('- '*10,'Optimise incl. EC',' -'*10)
                #print(res.keys()) # Prints available outputs for the object res
                if (res.success):
                    if (tPrint):
                        print('Nit =', res.nit) # Number of iterations
                else:
                    print('stat =', res.status)  # Optimisation status (int)
                    print('mess =', res.message) # Optimisation message
                # Check if optimised parameters satisfy the constraints
                check_constraints(res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],mu3)
                if (tPrint):
                    print('Px  =',res.x[0])
                    print('pk  = ({0}, {1}, {2})'.format(res.x[1],res.x[2],1 - res.x[1] - res.x[2]))
                    print('mu  = ({0}, {1}, {2})'.format(res.x[3],res.x[4],mu3))
                    print('SKL =',int(1.0 / res.fun))
                # Get final parameters from standard key length function
                SKL,QBERx,phi_x,nX,nZ,mX,lambdaEC,sX0,sX1,vz1,sZ1,mpn = key_length(res.x)
                # Store calculation parameters
                fulldata[count,:] = [ls+sysLoss,dt,int(1.0 / res.fun),QBERx,phi_x,
                                     nX,nZ,lambdaEC,sX0,sX1,vz1,sZ1,mpn,PolError,
                                     Pdc,Pap,NoPass,Rrate,eps_c,eps_s,res.x[0],
                                     res.x[1],res.x[2],1-res.x[1]-res.x[2],
                                     res.x[3],res.x[4],mu3,np.degrees(xi),
                                     min_elev,max_elev]
                # Store optimiser metrics
                optdata[count,:] = [res.fun,*res.x,res.optimality,
                                    res.constr_violation,*res.grad,
                                    *res.lagrangian_grad,res.nit,res.nfev,
                                    res.njev,res.nhev,res.cg_niter,res.tr_radius,
                                    res.constr_penalty,res.barrier_tolerance,
                                    res.barrier_parameter,res.status,
                                    res.cg_stop_cond]
                count += 1 # Increment calculation counter
                errcorrFunc = 'None' # Turn off error correction
                res = minimize(key_length_inv, x0, method='trust-constr',
                               jac=None, hess=None, hessp=None, bounds=bounds, 
                               constraints=lin_constraint, tol=None, callback=None, 
                               options={'xtol': xtol,
                                        'gtol': gtol, 
                                        'barrier_tol': btol,
                                        'sparse_jacobian': None, 
                                        'maxiter': Nmax,
                                        'verbose': 0, 
                                        'finite_diff_rel_step': None, 
                                        'initial_constr_penalty': const_pen,
                                        'initial_tr_radius': tr_rad, 
                                        'initial_barrier_parameter': barr_par, 
                                        'initial_barrier_tolerance': barr_tol, 
                                        'factorization_method': None,
                                        'disp': False})
                if (tPrint):
                    print('- '*10,'Optimise excl. EC',' -'*10)
                if (res.success):
                    if (tPrint):
                        print('Nit =', res.nit) # Number of iterations
                else:
                    print('stat =', res.status)  # Optimisation status (int)
                    print('mess =', res.message) # Optimisation message
                errcorrFunc = funcEC # Turn on error correction
                # Get final parameters from standard key length function
                SKL,QBERx,phi_x,nX,nZ,mX,lambdaEC,sX0,sX1,vz1,sZ1,mpn = key_length(res.x)
                # Check if optimised parameters satisfy the constraints
                check_constraints(res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],mu3)
                if (tPrint):
                    print('Px  =',res.x[0])
                    print('pk  = ({0}, {1}, {2})'.format(res.x[1],res.x[2],1 - res.x[1] - res.x[2]))
                    print('mu  = ({0}, {1}, {2})'.format(res.x[3],res.x[4],mu3))
                    print('SKL =',max(int(1.0 / res.fun-lambdaEC),0),
                          ' (',int(1.0 / res.fun),'-',int(lambdaEC),')')
                    print('-'*60,'\n')
    else:
        # Calculate the optimised SKL with the error correction term included
        # in the optimisation procedure.
        for ls in range(ls_range[0],ls_range[1]+ls_range[2],ls_range[2]):
            for dt in range(dt_range[0],dt_range[1]+dt_range[2],dt_range[2]):
                print('Calculation {0}: ls = {1}, dt = {2}'.format(count+1,ls,int(dt)))
                # Calculate optimised SKL
                res = minimize(key_length_inv, x0, method='trust-constr',
                               jac=None, hess=None, hessp=None, bounds=bounds, 
                               constraints=lin_constraint, tol=None, callback=None, 
                               options={'xtol': xtol,
                                        'gtol': gtol, 
                                        'barrier_tol': btol,
                                        'sparse_jacobian': None, 
                                        'maxiter': Nmax,
                                        'verbose': 0, 
                                        'finite_diff_rel_step': None, 
                                        'initial_constr_penalty': const_pen,
                                        'initial_tr_radius': tr_rad, 
                                        'initial_barrier_parameter': barr_par, 
                                        'initial_barrier_tolerance': barr_tol, 
                                        'factorization_method': None,
                                        'disp': False})

                #print(res.keys()) # Prints available outputs for the object res
                if (res.success):
                    if (tPrint):
                        print('Nit =', res.nit) # Number of iterations
                else:
                    print('stat =', res.status)  # Optimisation status (int)
                    print('mess =', res.message) # Optimisation message
                # Check if optimised parameters satisfy the constraints
                check_constraints(res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],mu3)
                if (tPrint):
                    print('Px  =',res.x[0])
                    print('pk  = ({0}, {1}, {2})'.format(res.x[1],res.x[2],1 - res.x[1] - res.x[2]))
                    print('mu  = ({0}, {1}, {2})'.format(res.x[3],res.x[4],mu3))
                    print('SKL =',int(1.0 / res.fun))
                    print('-'*60,'\n')
                # Get final parameters from standard key length function
                SKL,QBERx,phi_x,nX,nZ,mX,lambdaEC,sX0,sX1,vz1,sZ1,mpn = key_length(res.x)
                # Store calculation parameters
                fulldata[count,:] = [ls+sysLoss,dt,int(1.0 / res.fun),QBERx,phi_x,
                                     nX,nZ,lambdaEC,sX0,sX1,vz1,sZ1,mpn,PolError,
                                     Pdc,Pap,NoPass,Rrate,eps_c,eps_s,res.x[0],
                                     res.x[1],res.x[2],1-res.x[1]-res.x[2],
                                     res.x[3],res.x[4],mu3,np.degrees(xi),
                                     min_elev,max_elev]
                # Store optimiser metrics
                optdata[count,:] = [res.fun,*res.x,res.optimality,
                                    res.constr_violation,*res.grad,
                                    *res.lagrangian_grad,res.nit,res.nfev,
                                    res.njev,res.nhev,res.cg_niter,res.tr_radius,
                                    res.constr_penalty,res.barrier_tolerance,
                                    res.barrier_parameter,res.status,
                                    res.cg_stop_cond]
                count += 1 # Increment calculation counter
    tc1 = perf_counter() # Stop clock timer
    tp1 = process_time() # Stop CPU timer
else:
    # Calculate the SKL for the specified parameters (no optimisation).
    tc0 = perf_counter() # Start clock timer
    tp0 = process_time() # Start CPU timer
    count = 0 # Initialise calculation counter
    for ls in range(ls_range[0],ls_range[1]+ls_range[2],ls_range[2]):
        for dt in range(dt_range[0],dt_range[1]+dt_range[2],dt_range[2]):
            print('Calculation {0}: ls = {1}, dt = {2}'.format(count+1,ls,int(dt)))
            # Calculate the secure key length for the specified parameters.
            SKL,QBERx,phi_x,nX,nZ,mX,lambdaEC,sX0,sX1,vz1,sZ1,mpn = key_length(x0)
            if (tPrint):
                print('Px  =',x0[0])
                print('pk  = ({0}, {1}, {2})'.format(x0[1],x0[2],1 - x0[1] - x0[2]))
                print('mu  = ({0}, {1}, {2})'.format(x0[3],x0[4],mu3))
                print('SKL =', int(SKL))
                print('-'*60,'\n')
            # Store calculation parameters
            fulldata[count,:] = [ls+sysLoss,dt,SKL,QBERx,phi_x,
                                 nX,nZ,lambdaEC,sX0,sX1,vz1,sZ1,mpn,PolError,
                                 Pdc,Pap,NoPass,Rrate,eps_c,eps_s,x0[0],x0[1],
                                 x0[2],1-x0[1]-x0[2],x0[3],x0[4],mu3,
                                 np.degrees(xi),min_elev,max_elev]
            count += 1 # Increment calculation counter
    tc1 = perf_counter() # Stop clock timer
    tp1 = process_time() # Stop CPU timer

#******************************************************************************
# Print the calculation timings
#******************************************************************************
tc = tc1-tc0 # Calculation duration from clock
tp = tp1-tp0 # Calculation duration from CPU
if (not tPrint):
    print('')
print('Clock timer (s):',tc)
print('CPU timer (s):  ',tp,'\n')

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#******************************************************************************
#******************************************************************************
# Sort and output data
#******************************************************************************
#******************************************************************************

if (tWriteData):
    # Write out data in CSV format
    writeDataCSV(fulldata,outpath,outfile,header)
    if (tSortData):
        # Sort data by SKL per SysLoss
        sortdata = sort_data(fulldata,header,['SKL','SysLoss'],[True,False])
        if (tOptiData):
            # Only write out optimal SKL per SysLoss
            writeDataCSV(sortdata[::ndt,:],outpath,outfile+'_opt',header)
        else:
            # Write all sorted data
            writeDataCSV(sortdata,outpath,outfile+'_sort',header)
    if (tOptimise):
        # Write optimiser metrics
        writeDataCSV(optdata,outpath,'metrics',opt_head)