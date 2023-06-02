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

This version employs either the Chernoff or Hoeffding tail bounds for finite
block keys. It can also calculate the asymptotic key length (accumulated over
an infinite number of identical satellite overpasses).

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
from os.path import join

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

F_or_T = [False, True] # List used to switch between False or True values

#******************************************************************************
#******************************************************************************
#                      ---  USER INPUT SECTION (1)  ---
#     Select the type of secure key length calculation and intialise the
#     parameters
#******************************************************************************
#******************************************************************************

#******************************************************************************
# Select SKL calculation type via tOptimise
#******************************************************************************
#    True:  Optimise over the main protocol parameters.
#    False: Specify the main protocol parameters.
#******************************************************************************
tOptimise = F_or_T[0]  # False (0) or True (1)

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
    tInit = F_or_T[0]  # False (0) or True (1)

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
# Path to loss file  (empty = current directory)
# E.g. loss_path = 'C:\\path\\to\\directory'
loss_path = ''
# File containing loss data (for given xi value below)
loss_file = 'FS_loss_XI0.csv'
lc        = 3 # Column containing loss data in file (counting from 1)

#******************************************************************************
# Fixed system parameters
#******************************************************************************
# Angle between receiver zenith and satellite (from Earth's centre)
xi         = 0.0       # [rad.]
#mu3 = 10**(-9) # Weak coherent pulse 3 intensity, mu_3
mu3        = 0         # Intensity of pulse 3 (fixed)
# Prescribed errors in protocol correctness and secrecy
eps_c      = 10**(-15) # Correctness parameter
eps_s      = 10**(-9)  # Secrecy parameter
# Intrinsic Quantum Bit Error Rate (QBER_I)
QBERI_list = [0.001,0.003,0.005] # list, array, tuple or singleton
# Extraneous count probability
Pec_list   = [1e-8,1e-7,1e-6] # list, array, tuple or singleton
# Afterpulse probability
Pap        = 0.001     # After-pulse probability
# Number of satellite passes
NoPass     = 1         # Number of satellite passes
# Repetition rate of the source in Hz
Rrate      = 1*10**(9) # Source rate (Hz)
# Number of pulses sent in Hz
Npulse     = NoPass*Rrate

#******************************************************************************
# Define (inclusive) range for looped parameters: dt and ls
#******************************************************************************
# Index for windowing time slot arrays, e.g. A(t)[t0-dt:t0+dt], with dt <= 346
dt_range = np.array([200, 350, 10]) # Start, stop, step index (defualt)
# Set a minimum elevation angle for transmission, this value will override
# the values corresponding to dt_range[1] if necessary.
min_elev    = 10.0 # Minimum elevation transmission angle (degs)
shift_elev0 = 0.0  # Shift the elevation angle taken as t = 0 (degs).

# Excess system loss in dB
ls_range = np.array([0, 12, 2])   # Start, stop, step value

#******************************************************************************
# NOTE: All nominal losses have been moved into FS_loss_XI<xi>.csv for now and 
# so we simply set the efficiencies listed below to unity.
#******************************************************************************
# Detector efficiency
eta_d = 1.0           # Typically 40-60%
# Internal telescope transmitter loss
eta_transmitter = 1.0 # Typically ~50%
# Combined losses
eta = eta_d * eta_transmitter

#******************************************************************************
# Output file options
#******************************************************************************
# Write output to CSV file?
tFullData  = F_or_T[1] # False (0) or True (1)
# Write out only max values of SKL for dt?
tOptiData  = F_or_T[1] # False (0) or True (1)
# Write out optimised values for each system in one file?
tMultiOpt  = F_or_T[1] # False (0) or True (1)
# Write out optimiser metrics for each (final) calculation?
tMetrics   = F_or_T[1] # False (0) or True (1)
# Path for output files (empty = current directory)
# E.g. outpath = 'C:\\path\\to\\directory'
outpath    = ''    
# Basename for output file (excluding .csv)
outbase    = "out"

# Print values to StdOut during calculations?
tPrint     = F_or_T[1] # False (0) or True (1)

#******************************************************************************
# Advanced/additional parameters
#******************************************************************************
# Use the Chernoff bounds when estimating statistical fluctuations in the
# count statistics? Otherwise use the Hoeffding bound.
#tChernoff = F_or_T[1]  # False (0) or True (1)
# Select the type of tail bounds to use for estimating statistical fluctuations
# in the count statistics.
#   'Chernoff'   = boundFunc[0] => Different upper and lower bound terms
#   'Hoeffding'  = boundFunc[1] => Same upper and lower bound terms
#   'Asymptotic' = boundFunc[2] => No tail bounds. This choice forces also 
#                                  errcorrFunc = 'block' below.
boundOpts = ['Chernoff','Hoeffding','Asymptotic']
boundFunc = boundOpts[0] # Select an option from the list above.

# Select the method for estimating the number of bits sacrificed for error
# correction, listed below in order of decreasing precision (increasing
# smoothness).
#   'logM'  = logM(nX, QBERx, eps_c) = errcorrOpts[0]
#   'block' = 1.16 * nX * h(QBERx)   = errcorrOpts[1]
#   'mXtot' = 1.16 * mXtot           = errcorrOpts[2]
#   'None'  = 0                      = errcorrOpts[3]
errcorrOpts = ['logM','block','mXtot','None']
errcorrFunc = errcorrOpts[0] # Select a method from the list above.

# Compare SKL optimised with/without EC (for optimised calculation only).
tCompareEC  = F_or_T[0]  # False (0) or True (1)

# Numerical value to use when denominator values are potentially zero
num_zero = 10**(-10) # Note to self: Use num_min below?

opt_methods = ['COBYLA','SLSQP','trust-constr']
method      = opt_methods[0] # Select a optimisation method
NoptMin     = 10        # Minimum No. of optimisations to strive for
NoptMax     = 1000      # Maximum No. of optimisations (not used)
tStopZero   = F_or_T[1] # Stop optimizing if the first NoptMin return SKL = 0?
tStopBetter = F_or_T[1] # Stop after NoptMin optimizations if SKL improved?

if method == 'trust-constr':
    # Optimiser options: minimize, method='trust-constr'
    xtol      = 1.0e-8  # Tolerance to terminate by change of independent variable(s)
    gtol      = 1.0e-10 # Tolerance to terminate Lagrangian gradient
    btol      = 1.0e-8  # Threshold on the barrier parameter for termination
    Nmax      = 1000    # Max No. of iterations
    const_pen = 1.0     # Initial constraint penalty parameter (default = 1.0)
    tr_rad    = 1.0     # Initial trust radius (default = 1.0)
    barr_par  = 0.1     # Initial barrier parameter (default = 0.1)
    barr_tol  = 0.1     # Initial barrier tolerance for barrier sub- (default = 0.1)
elif method == 'COBYLA':
    # Optimiser options: minimize, method='COBYLA'
    Nmax   = 1000       # Max No. of iterations
    ctol   = 1.0e-12    # Constraint absolute tolerance.
    rhobeg = 0.002      # Reasonable initial changes to the variables.
elif method == 'SLSQP':
    # Optimiser options: minimize, method='SLSQP'
    Nmax = 1000         # Max No. of iterations
    ftol = 1.0e-12      # Precision goal for the value of f in the stopping criterion.
    eps  = 1.0e-7       # Step size used for numerical approximation of the Jacobian.
else:
    print('Optimiser method not recognised:',method)
    print('Select one of:',*["'{0}'".format(m) for m in opt_methods])
    exit(1)

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
cvs = np.loadtxt(join(loss_path, loss_file), delimiter= ',', skiprows=1, 
                 usecols=(0, 1, lc-1,))

# Free space loss in dB (to be converted to efficiency)
# Returns the (t,) array FSeff, where t is the total number of time-slots
FSeff = cvs[:,2]

# Find the time slot at the centre of the pass where t = 0.
time0pos   = np.where(cvs[:,0] == 0)[0][0]
time0elev  = cvs[time0pos,1] # Elevation angle at t = 0 (rads).
time0shift = time0pos # Take a temporary copy to use later.

# Nominal system loss: based on zenith coupling efficiency and nominal losses
sysLoss = -10*(np.math.log10(FSeff[time0pos]) + np.math.log10(eta))

# Maximum elevation angle (degs) of satellite pass
#max_elev = np.degrees(cvs[time0pos,1])
max_elev = np.degrees(time0elev)

# Keep elevation shift within bounds
#shift_elev0 = min(90.0,max(0.0,shift_elev0))
# Check that elevation shift angle specified is valid
if (shift_elev0 < 0.0 or (not shift_elev0 < 90)):
    print('Error! Shift elevation angle for t = 0 out of bounds:',shift_elev0)
    print('Angle should be >= 0 and < 90 degrees.')
    exit(1)

if (shift_elev0 != 0.0):
    # Shift the elevation angle taken as t = 0 away from elev = 90 deg.
    # Find the first array index for an elevation angle greater than, or equal
    # to, the shifted angle requested.
    time0pos   = np.where(cvs[:,1] >= (time0elev - np.radians(shift_elev0)))[0][0]
    time0elev  = cvs[time0pos,1] # New elevation angle at t = 0 (rads).
    time0shift = abs(time0pos - time0shift) # Shift in time slots between old and new t = 0.
else:
    # No shift requested, t = 0 is at elev = 90 deg.
    time0shift = 0 # Reset this value to zero.

#******************************************************************************
#******************************************************************************
# Parameter checks
#******************************************************************************
#******************************************************************************
# Flag that controls if any data files are to be written
tWriteFiles = False
if any([tFullData,tOptiData,tMultiOpt,tMetrics]):
    tWriteFiles = True

if tOptimise:
    # Sanity check on parameter bounds
    xb[xb < 0] = 0 # All bounds must be positive
    xb[xb > 1] = 1 # All bounds are less than or equal to 1
    if errcorrFunc in ['None','none']:
        tCompareEC = False # Don't perform the same calculation twice

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
dt_max = int(0.5*(len(FSeff) - 1) - time0shift) # Maximum time window half-width
dt_range[dt_range < 0]       = 0       # All values must be positive
dt_range[dt_range > dt_max]  = dt_max  # Max cannot exceed No. of time-slots
dt_range[dt_range > dt_elev] = dt_elev # Limit range by minimum elevation value

# Get minimum elevation for transmission (possibly greater than value specified)
minElevpos = np.where(cvs[:,0] <= dt_range[1])[0][0] # Index of first value
min_elev   = np.degrees(cvs[minElevpos,1]) # Minimum elevation (degs)

# Ensure asymptotic case uses correct error estimation function, etc.
if boundFunc in ['Asymptotic','asymptotic']:
    Npulse = Rrate / NoPass # Rescale No of pulses to per-pass
    NoPass = 1              # Actually infinite but set to unity
    errcorrFunc = 'block'   # Asymptotic error function
    
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

def x0_rand(mu3,xb,num_min):
    """
    Randomly initialise the 5 protocol parameters using the specified bounds.
    Parameters and bounds should be specified in the order {Px,pk1,pk2,mu1,mu2}.

    Parameters
    ----------
    mu3 : float
        Intensity of pulse 3 (vacuum).
    xb : float, array-like
        Upper and lower bounds for the protocol parameters. (5,2)
    num_min : float
        An arbitrarily small number.

    Returns
    -------
    x0 : float, array
        Randomly initialised protocol parameters.

    """
    Px_i  = np.random.rand() * (xb[0,1] - xb[0,0] - 2*num_min) + xb[0,0] + \
            num_min
    pk1_i, pk2_i = 1.0, 1.0
    while (pk1_i+pk2_i >= 1.0):
        pk1_i = np.random.rand() * (xb[1,1] - xb[1,0] - 2*num_min) + \
                xb[1,0] + num_min
        pk2_i = np.random.rand() * (min(xb[2,1],1-pk1_i) - xb[2,0] - \
                                    2*num_min) + xb[2,0] + num_min
    mu1_i = np.random.rand() * (xb[3,1] - max(xb[3,0],2*mu3) - 2*num_min) + \
            max(xb[3,0],2*mu3) + num_min
    mu2_i = np.random.rand() * (min(xb[4,1],mu1_i) - max(xb[4,0],mu3) - \
                                2*num_min) + max(xb[4,0],mu3) + num_min
    return np.array([Px_i,pk1_i,pk2_i,mu1_i,mu2_i])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def getOptData(Nopt,Ntot,x0,res,method):
    """
    Returns a list of output metrics from the scipy.optimize results object res.

    Parameters
    ----------
    Nopt : integer
        Number of optimisations performed.
    Ntot : integer
        Total number of function evaluations.
    x0 : float, array-like
        Initial protocol parameters
    res : object, dictionary
        Optimisation results.
    method : string
        Optimization method.

    Returns
    -------
    optData : list
        List of optimisation metrics and data.

    """
    if method == 'trust-constr':
        # 'Nopt' 'Ntot' 'x0', 'x', 'fun', 'status', 'success', 'nfev', 'njev',
        # 'nhev', 'nit', 'grad', 'lagrangian_grad', 'cg_niter', 'cg_stop_cond',
        # 'constr_violation', 'constr_penalty',  'tr_radius',  'niter',
        # 'barrier_parameter', 'barrier_tolerance', 'optimality',
        # 'execution_time'
        return [Nopt,Ntot,*x0,*res.x,res.fun,res.status,res.success,res.nfev,
                res.njev,res.nhev,res.nit,*res.grad,*res.lagrangian_grad,
                res.cg_niter,res.cg_stop_cond,res.constr_violation,
                res.constr_penalty,res.tr_radius,res.niter,
                res.barrier_parameter,res.barrier_tolerance,
                res.optimality,res.execution_time]
    elif method == 'COBYLA':
        # 'Nopt' 'Ntot' 'x0', 'x', 'fun', 'status', 'success', 'nfev', 'maxcv'
        return [Nopt,Ntot,*x0,*res.x,res.fun,res.status,res.success,res.nfev,
                res.maxcv]
    elif method == 'SLSQP':
        # 'Nopt' 'Ntot' 'x0', 'x', 'fun', 'status', 'success', 'nfev', 'njev',
        # 'nit'
        return [Nopt,Ntot,*x0,*res.x,res.fun,res.status,res.success,res.nfev,
                res.njev,res.nit]
    else:
        return []

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def writeDataCSV(data,outpath,outfile,out_head=None,message='data'):
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
    message : string, optional
        Data description for print command, default = 'data'.
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
    filename = join(outpath, outfile + '.csv')
    print('Saving',message,'in file:',filename)
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
    tags     = header.split(',') # Split the header string into separate
                                 # elements of a list.
    nsort    = len(sort_tags)    # Number of columns to sort by
    for ii in range(0,nsort,1):
        try:
            sc = tags.index(sort_tags[ii]) # Check tags match headings
        except ValueError:
            print('Error! Sort tag not recognised:',sort_tags[0])
            return None
    if (len(sort_tags) != len(rev_sort)):
        print("Error! Lists 'sort_tags' and 'rev_sort' have different lengths")
        print('len(sort_tags) =',len(sort_tags),' len(rev_sort) =',
              len(rev_sort))
        return None
    sc       = tags.index(sort_tags[0]) # First column to sort by
    sortdata = data # Take a copy to change it
    if rev_sort[0]:
        # Reverse sort data array
        sortdata = sortdata[sortdata[:,sc].argsort()[::-1]] # sort by column sc
    else:
        # Sort data array
        sortdata = sortdata[sortdata[:,sc].argsort()] # sort by column sc
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
    mu = np.array([x[3],x[4],mu3])
    # Probability of Alice sending a pulse of intensity mu_k
    P  = np.array([x[1],x[2],1 - x[1] - x[2]])
    
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
    if boundFunc in ['Asymptotic','asymptotic']:
        # Asymptotic phase error rate for single photon events in the sifted X
        # basis
        phi_x = min(ratio, 0.5)
        # Secret key length in the asymptotic regime
        l = max((sx0 + sx1 * (1 - h(phi_x)) - lambdaEC) * 
             heaviside(mu[0] - mu[1] - mu[2]) * heaviside(P[2]), 0.0)
    else:
        # Phase error rate for single photon events in the sifted X basis
        phi_x = min(ratio + gamma(eps_s,ratio,sz1,sx1), 0.5)
        # Secret key length in the finite regime
        l = max((sx0 + sx1 * (1 - h(phi_x)) - lambdaEC -
             6*np.math.log(21.0 / eps_s, 2) - np.math.log(2.0 / eps_c, 2)) * 
             heaviside(mu[0] - mu[1] - mu[2]) * heaviside(P[2]), 0.0)
        l = l / NoPass # Normalise by the number of satellite passes used
        
    return l, QBERx, phi_x, nX, nZ, mXtot, lambdaEC, sx0, sx1, vz1, sz1, mpn

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Secure Key Length (SKL) inverse function - returns (1 / SKL) - for optimizer
def key_length_inv(x):
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
        #return l        # Negative key length, NaN? --Useful for troubleshooting
        return num_max  # ~1/0 --Stops calculations grinding to a halt

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#******************************************************************************
#******************************************************************************
# Secure key length calculation
#******************************************************************************
#******************************************************************************

#******************************************************************************
# Initialise calculation parameters
#******************************************************************************
if (tOptimise and not tInit):
    # Randomly assign the initial protocol parameters
    x0 = x0_rand(mu3,xb,num_min)
else:
    # Store initial/fixed protocol parameters as an array
    x0 = np.array([Px_i,pk1_i,pk2_i,mu1_i,mu2_i])

# Check initial parameters are within the bounds and constraints of the 
# protocol.
check_constraints(x0[0],x0[1],x0[2],x0[3],x0[4],mu3)

#******************************************************************************
# Initialise loop ranges
#******************************************************************************
# Set the number of outputs to store
if (ls_range[0] == ls_range[1] or ls_range[2] == 0):
    nls = 1
else:
    nls = int((ls_range[1] - ls_range[0]) / float(ls_range[2])) + 1
if (dt_range[0] == dt_range[1] or dt_range[2] == 0):
    ndt = 1
else:
    ndt = int((dt_range[1] - dt_range[0]) / float(dt_range[2])) + 1
    
npc = len(Pec_list)   # Number of Pec values to calculate for
nqi = len(QBERI_list) # Number of QBERI values to calculate for

x0i = np.empty((5,npc,nqi,nls,ndt)) # Array to store initialisation parameters
ci  = np.empty((4,),dtype=np.int16) # Array to count the various loops

#******************************************************************************
# Initialise output data storage and headers
#******************************************************************************
# Header for CSV file: Data columns
header = "SysLoss,dt,SKL,QBERx,phiX,nX,nZ,lambdaEC,sX0,sX1,vZ1,sZ1," + \
         "mean photon no.,QBERI,Pec,Pap,NoPass,Rrate,eps_c,eps_s," + \
         "Px,P1,P2,P3,mu1,mu2,mu3,xi (deg),minElev (deg),maxElev (deg)," + \
         "shiftElev (deg)"
# Initialise a data storage array: shape(No. of data runs, No. of metrics)   
fulldata = np.empty((nls*ndt,len(header.split(","))))

if (tOptimise):
    # Header for CSV file: Optimiser metrics
    if method == 'trust-constr':
        opt_head = "Nopt,Ntot,x0i,x1i,x2i,x3i,x4i,x0,x1,x2,x3,x4,1/fun," + \
                   "status,success,nfev,njev,nhev,Nit,grad0,grad1,grad2," + \
                   "grad3,grad4,lg_gr0,lg_gr1,lg_gr2,lg_gr3,lg_gr4,Ncg," + \
                   "cg_stop,con_vln,con_pen,tr_rad,niter,barr_par," + \
                   "barr_tol,opt,ex_time"
    elif method == 'COBYLA':
        opt_head = "Nopt,Ntot,x0i,x1i,x2i,x3i,x4i,x0,x1,x2,x3,x4,1/fun," + \
                   "status,success,nfev,maxcv"
    elif method == 'SLSQP':
        opt_head = "Nopt,Ntot,x0i,x1i,x2i,x3i,x4i,x0,x1,x2,x3,x4,1/fun," + \
                   "status,success,nfev,njev,Nit"
    else:
        opt_head == ""
    # Initialise a data storage array for optimiser metrics
    optdata = np.empty((nls*ndt,len(opt_head.split(","))))
    
if tMultiOpt:
    # Initialise a storage array for multi-system optimal data
    multidata = np.empty((nls*npc*nqi,len(header.split(","))))

#******************************************************************************
# Print main calculation parameters
#******************************************************************************
print('_'*60,'\n')
print('-'*60)
print('SatQuMA v1.1 (prototype-monolith)')
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
print('Using {} bounds for statistical fluctuations'.format(boundFunc))
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
if (nls*ndt*npc*nqi > 1):
    print('User requested',nls*ndt*npc*nqi,'calculations:')
    print(' >',ndt,'transmission windows')
    print(' >',nls,'system loss values')
    print(' >',npc,'extraneous count rates (Pec)')
    print(' >',nqi,'intrinsic QBERs (QBERI)')
else:
    print('User requested',nls*ndt*npc*nqi,'calculation')
    print(' >',ndt,'transmission window')
    print(' >',nls,'system loss value')
    print(' >',npc,'extraneous count rate (Pec)')
    print(' >',nqi,'intrinsic QBER (QBERI)')
print('-'*60)
print('_'*60,'\n')

if tPrint:
    print('Reading losses from file:',loss_path+loss_file)
    print('-'*60)

#******************************************************************************
# Begin main calculations
#******************************************************************************
if (tOptimise):   
    # Calculate the secure key length by optimising over the protocol
    # parameters.
    # Build bounds arrays for x = [Px,pk1,pk2,mu1,mu2]
    # Lower bounds for parameters
    lb = np.array([xb[0,0],xb[1,0],xb[2,0],xb[3,0],xb[4,0]]) + num_min
    # Upper bounds for parameters
    ub = np.array([xb[0,1],xb[1,1],xb[2,1],xb[3,1],xb[4,1]]) - num_min
    from scipy.optimize import Bounds
    # Store the parameters bounds as an object, as required by minimize() for
    # method='trust-constr'
    bounds = Bounds(lb, ub)          # Store upper and lower bounds
    if method == 'trust-constr':
        # Build linear constraint arrays for x = [Px,pk1,pk2,mu1,mu2]^T such
        # that: (lb_l)^T <= A_l x <= (ub_l)^T
        # Constraints with bounds = +/-np.inf are interpreted as one-sided
        lb_l = np.array([xb[0,0],num_min,xb[2,0],(mu3 + num_min),
                         (mu3 + num_min)])
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
        # Store the linear constraints as an object, as required by minimize()
        # for method='trust-constr'
        cons = (LinearConstraint(A_l, lb_l, ub_l),)
        # Set specific optimiser options
        options = {'xtol': xtol,
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
                   'disp': False}
    else:
        # Build inequality constraint dictionary for 'COBYLA' or 'SLSQP'
        cons_type = 'ineq' # C_j[x] >= 0
        def cons_fun(x):
            # Linear constraint inequality function:
            #   (1)         1 - pk1 - pk2 >= 0,
            #   (2) mu1 - mu2 - mu3 - eps >= 0,
            #   (3)       mu2 - mu3 - eps >= 0,
            # where eps is an arbitrarily small number
            return np.array([1 - x[1] - x[2],
                             x[3] - x[4] - mu3 - num_min,
                             x[4] - mu3 - num_min])
        def cons_jac(x):
            # Jacobian of the linear constraint inequality function
            return np.array([[0,-1,-1,0,0],
                             [0,0,0,1,-1],
                             [0,0,0,0,1]])
        # Define linear constraint dictionary
        lin_cons = {'type' : cons_type,
                    'fun'  : cons_fun,
                    'jac'  : cons_jac}
        if method == 'COBYLA':
            bounds = None # COBYLA doesn't support bounds
            # Build inequality constraint dictionaries for bounds
            cons_type = 'ineq' # C_j[x] >= 0
            def upper_fun(x):
                # Upper bound inequality function
                return np.array([ub[0] - x[0],
                                 ub[1] - x[1],
                                 ub[2] - x[2],
                                 ub[3] - x[3],
                                 ub[4] - x[4]])
            def upper_jac(x):
                # Upper bound inequality Jacobian
                return np.array([[-1,0,0,0,0],
                                 [0,-1,0,0,0],
                                 [0,0,-1,0,0],
                                 [0,0,0,-1,0],
                                 [0,0,0,0,-1]])
            # Define upper bound inequality dictionary
            upper = {'type' : cons_type,
                     'fun'  : upper_fun,
                     'jac'  : upper_jac}
            def lower_fun(x):
                # Lower bound inequlity function
                return np.array([x[0] - lb[0],
                                 x[1] - lb[1],
                                 x[2] - lb[2],
                                 x[3] - lb[3],
                                 x[4] - lb[4]])
            def lower_jac(x):
                # Lower bound inequality Jacobian
                return np.array([[1,0,0,0,0],
                                 [0,1,0,0,0],
                                 [0,0,1,0,0],
                                 [0,0,0,1,0],
                                 [0,0,0,0,1]])
            # Define lower bound inequality dictionary
            lower = {'type' : cons_type,
                     'fun'  : lower_fun,
                     'jac'  : lower_jac}
            # Tuple of all bounds
            cons = (upper, lower, lin_cons)
            # Set specific optimiser options
            options = {'rhobeg': rhobeg,
                       'maxiter': Nmax, 
                       'disp': False, 
                       'catol': ctol}
        elif method == 'SLSQP':
            # Singleton tuple of bounds
            cons = (lin_cons,)
            # Set specific optimiser options
            options = {'maxiter': Nmax,
                       'ftol': ftol,
                       'iprint': 1,
                       'disp': False,
                       'eps': eps,
                       'finite_diff_rel_step': None}
        else:
            print('Optimiser method not recognised:',method)
            print('Select one of:',*["'{0}'".format(m) for m in opt_methods])
            exit(1)

    # Print out the intial parameters, bounds, and algorithm options
    if tPrint:
        print('Initial protocol parameters:')
        print('Px_i =',x0[0])
        print('pk_i = (',x0[1],x0[2],1 - x0[1] - x0[2],')')
        print('mu_i = (',x0[3],x0[4],mu3,')')
        print('-'*60)
        print('Protocol parameter bounds (lower,upper):')
        print('Px   = ({0}, {1})'.format(lb[0],ub[0]))
        print('pk1  = ({0}, {1})'.format(lb[1],ub[1]))
        print('pk2  = ({0}, {1})'.format(lb[2],ub[2]))
        print('mu1  = ({0}, {1})'.format(lb[3],ub[3]))
        print('mu2  = ({0}, {1})'.format(lb[4],ub[4]))
        print('-'*60)
        print("Optimiser parameters ('{0}'):".format(method))
        print('NoptMin =',NoptMin)
        for key, value in options.items():
            print("{0} = {1}".format(key, value))
        print('-'*60)
        print('-'*60,'\n')
        
tc00 = perf_counter() # Start clock timer
tp00 = process_time() # Start CPU timer

count = 0 # Initialise calculation counter
ci[0] = 0      
for Pec in Pec_list:
    ci[1] = 0
    for QBERI in QBERI_list:
        outfile = outbase + '_Pec_{}_QBERI_{}_{}GHz'.format(ci[0],ci[1],Rrate/1.0e9)
        tc0 = perf_counter() # Start clock timer
        tp0 = process_time() # Start CPU timer
        if tOptimise:
            from scipy.optimize import minimize      
            # Perform the optimization of the secure key length
            if tCompareEC:
                # Compare optimised SKL for, first, including error correction during
                # the optimissation procedure and then, second, for including the
                # error correction term after the optimisation procedure.
                funcEC = errcorrFunc # Store error correction function choice
                ci[2] = 0
                for ls in range(ls_range[0],ls_range[1]+1,ls_range[2]):
                    ci[3] = 0
                    for dt in range(dt_range[0],dt_range[1]+1,dt_range[2]):
                        print('Calculation {}: Pec = {:5.2e}, QBERI = {:5.2e}, ls = {}, dt = {}'.format(
                            count+1,Pec,QBERI,ls,int(dt)))
                        if (tPrint):
                            print('- '*10,'Optimise incl. EC',' -'*10)
                        # Re-set initial parameters (if required)
                        if tInit:
                            # Store initial/fixed protocol parameters as an array
                            x0 = np.array([Px_i,pk1_i,pk2_i,mu1_i,mu2_i]) 
                        else:
                            if ci[2] > 0:
                                # Use the optimised parameters from the previous shift angle calculation
                                # as the initial values for this calculation
                                x0 = x0i[:,ci[0],ci[1],ci[2]-1,ci[3]]
                            elif ci[3] > 0:
                                # Use the optimised parameters from the previous calculation
                                # as the initial values for this calculation
                                x0 = x0i[:,ci[0],ci[1],ci[2],ci[3]-1]
                        # Calculate optimised SKL
                        res = minimize(key_length_inv,x0,args=(),method=method,
                                       jac=None,hess=None,hessp=None,bounds=bounds, 
                                       constraints=cons,tol=None,callback=None, 
                                       options=options)
                        Ntot = res.nfev # Initilaise total No. of function evaluations
                        Nopt = 1        # Number of optimisation calls
                        # Re-run optimization until Nmax function evaluations
                        # have been used. Take a copy of initial results to compare.
                        x0c  = x0
                        SKLc = int(1.0 / res.fun)
                        resc = res
                        Nzero = 0 # Number of times we get SKL == 0
                        while Ntot < Nmax or Nopt < NoptMin:
                            Nopt += 1
                            # Randomly assign new initial parameters
                            x0 = x0_rand(mu3,xb,num_min)
                            # Calculate optimised SKL
                            res = minimize(key_length_inv,x0,args=(),method=method,
                                           jac=None,hess=None,hessp=None,bounds=bounds, 
                                           constraints=cons,tol=None,callback=None, 
                                           options=options)
                            if int(1.0 / res.fun) > 0:
                                if int(1.0 / res.fun) > SKLc:
                                    if Nopt >= NoptMin and tStopBetter:
                                        break # A better value was found!
                                    else:
                                        # Store new set of best parameters
                                        x0c  = x0
                                        resc = res
                                        SKLc = int(1.0 / res.fun)
                                else:
                                    # Reset to the best parameters
                                    x0  = x0c
                                    res = resc
                            else:
                                # SKL = 0. Reset to the 'best' parameters,
                                # (may still give SKL = 0).
                                Nzero += 1
                                if Nopt > NoptMin:
                                    if Nzero / (Nopt - 1) == 1:
                                        # We get SKL = 0 every time.
                                        if tStopZero:
                                            break
                                x0  = x0c
                                res = resc
                            Ntot += res.nfev
                        print('Nopt =',Nopt)
                        #print('Ntot =',Ntot)
                        #print(res.keys()) # Prints available outputs for the object res
                        if (res.success):
                            if (tPrint):
                                if method in ['trust-constr','SLSQP']:
                                    print('Nit  =', res.nit) # Number of iterations
                                else:
                                    print('Nit  =', res.nfev) # Number of iterations
                        else:
                            print("Optimiser status = {}: {}".format(res.status,
                                                                       res.message))
                        # Check if optimised parameters satisfy the constraints
                        check_constraints(res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],
                                          mu3)
                        if (tPrint):
                            print('Px   =',res.x[0])
                            print('pk   = ({}, {}, {})'.format(res.x[1],res.x[2],
                                                                 1 - res.x[1]
                                                                 - res.x[2]))
                            print('mu   = ({}, {}, {})'.format(res.x[3],res.x[4],mu3))
                            print('SKL  = {:e}'.format(int(1.0 / res.fun)))
                        # Get final parameters from standard key length function
                        SKL,QBERx,phi_x,nX,nZ,mX,lambdaEC,sX0,sX1,vz1,sZ1,mpn = \
                            key_length(res.x)
                        # Store calculation parameters
                        fulldata[ci[2]*ndt + ci[3],:] = [ls+sysLoss,dt,int(1.0 / res.fun),QBERx,
                                             phi_x,nX,nZ,lambdaEC,sX0,sX1,vz1,sZ1,mpn,
                                             QBERI,Pec,Pap,NoPass,Rrate,eps_c,eps_s,
                                             res.x[0],res.x[1],res.x[2],1-res.x[1]
                                             -res.x[2],res.x[3],res.x[4],mu3,
                                             np.degrees(xi),min_elev,max_elev,
                                             shift_elev0]
                        # Store optimiser metrics
                        optdata[ci[2]*ndt + ci[3],:] = getOptData(Nopt,Ntot,x0,res,method)
                        count += 1 # Increment calculation counter
                        if (tPrint):
                            print('- '*10,'Optimise excl. EC',' -'*10)
                        errcorrFunc = 'None' # Turn off error correction
                        # Calculate optimised SKL
                        res = minimize(key_length_inv, x0, args=(), method=method,
                                       jac=None, hess=None, hessp=None, bounds=bounds, 
                                       constraints=cons, tol=None, callback=None, 
                                       options=options)
                        if (res.success):
                            if (tPrint):
                                if method in ['trust-constr','SLSQP']:
                                    print('Nit  =', res.nit) # Number of iterations
                                else:
                                    print('Nit  =', res.nfev) # Number of iterations
                        else:
                            #print('stat =', res.status)  # Optimisation status (int)
                            #print('mess =', res.message) # Optimisation message
                            print("Optimiser status = {}: {}".format(res.status,
                                                                       res.message))
                        errcorrFunc = funcEC # Turn on error correction
                        # Get final parameters from standard key length function
                        SKL,QBERx,phi_x,nX,nZ,mX,lambdaEC,sX0,sX1,vz1,sZ1,mpn = \
                            key_length(res.x)
                        # Check if optimised parameters satisfy the constraints
                        check_constraints(res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],
                                          mu3)
                        if (tPrint):
                            print('Px   =',res.x[0])
                            print('pk   = ({}, {}, {})'.format(res.x[1],res.x[2],
                                                                 1 - res.x[1]
                                                                 - res.x[2]))
                            print('mu   = ({}, {}, {})'.format(res.x[3],res.x[4],
                                                                 mu3))
                            print('SKL  =',max(int(1.0 / res.fun-lambdaEC),0),
                                  ' (',int(1.0 / res.fun),'-',int(lambdaEC),')')
                            print('SKL  = {:e} ({:e} - {:e})'.format(
                                max(int(1.0 / res.fun-lambdaEC),0),
                                int(1.0 / res.fun),int(lambdaEC)))
                            print('-'*60,'\n')
                        # Store protocol parameters to initialise calculations
                        if np.isnan(int(1.0 / res.fun)) or np.isinf(int(1.0 / res.fun)):
                            x0i[:,ci[0],ci[1],ci[2],ci[3]] = x0_rand(mu3,xb,num_min)
                        else:
                            if int(1.0 / res.fun) > 0:
                                x0i[:,ci[0],ci[1],ci[2],ci[3]] = res.x
                            else:
                                x0i[:,ci[0],ci[1],ci[2],ci[3]] = x0_rand(mu3,xb,num_min)
                        ci[3] += 1 # dt loop counter
                    ci[2] += 1 # ls loop counter
            else:
                # Calculate the optimised SKL with the error correction term included
                # in the optimisation procedure.
                ci[2] = 0
                for ls in range(ls_range[0],ls_range[1]+1,ls_range[2]):
                    ci[3] = 0
                    for dt in range(dt_range[0],dt_range[1]+1,dt_range[2]):
                        print('Calculation {}: Pec = {:5.2e}, QBERI = {:5.2e}, ls = {}, dt = {}'.format(
                            count+1,Pec,QBERI,ls,int(dt)))
                        # Re-set initial parameters (if required)
                        if tInit:
                            # Store initial/fixed protocol parameters as an array
                            x0 = np.array([Px_i,pk1_i,pk2_i,mu1_i,mu2_i]) 
                        else:
                            if ci[2] > 0:
                                # Use the optimised parameters from the previous shift angle calculation
                                # as the initial values for this calculation
                                x0 = x0i[:,ci[0],ci[1],ci[2]-1,ci[3]]
                            elif ci[3] > 0:
                                # Use the optimised parameters from the previous calculation
                                # as the initial values for this calculation
                                x0 = x0i[:,ci[0],ci[1],ci[2],ci[3]-1]
                        # Calculate optimised SKL
                        res = minimize(key_length_inv,x0,args=(),method=method,
                                       jac=None,hess=None,hessp=None,bounds=bounds, 
                                       constraints=cons,tol=None,callback=None, 
                                       options=options)
                        Ntot = res.nfev # Initilaise total No. of function evaluations
                        Nopt = 1        # Number of optimisation calls
                        # Re-run optimization until Nmax function evaluations
                        # have been used. Take a copy of initial results to compare.
                        x0c  = x0
                        SKLc = int(1.0 / res.fun)
                        resc = res
                        Nzero = 0 # Number of times we get SKL == 0
                        while Ntot < Nmax or Nopt < NoptMin:
                            Nopt += 1
                            # Randomly assign new initial parameters
                            x0 = x0_rand(mu3,xb,num_min)
                            # Calculate optimised SKL
                            res = minimize(key_length_inv,x0,args=(),method=method,
                                           jac=None,hess=None,hessp=None,bounds=bounds, 
                                           constraints=cons,tol=None,callback=None, 
                                           options=options)
                            if int(1.0 / res.fun) > 0:
                                if int(1.0 / res.fun) > SKLc:
                                    if Nopt >= NoptMin and tStopBetter:
                                        break # A better value was found!
                                    else:
                                        # Store new set of best parameters
                                        x0c  = x0
                                        resc = res
                                        SKLc = int(1.0 / res.fun)
                                else:
                                    # Reset to the best parameters
                                    x0  = x0c
                                    res = resc
                            else:
                                # SKL = 0. Reset to the 'best' parameters,
                                # (may still give SKL = 0).
                                Nzero += 1
                                if Nopt > NoptMin:
                                    if Nzero / (Nopt - 1) == 1:
                                        # We get SKL = 0 every time.
                                        if tStopZero:
                                            break
                                x0  = x0c
                                res = resc
                            Ntot += res.nfev
                        print('Nopt =',Nopt)
                        #print('Ntot =',Ntot)
                        if (res.success):
                            if (tPrint):
                                if method in ['trust-constr','SLSQP']:
                                    print('Nit  =', res.nit) # Number of iterations
                                else:
                                    print('Nit  =', res.nfev) # Number of iterations
                        else:
                            print("Optimiser status = {}: {}".format(res.status,
                                                                       res.message))
                        # Check if optimised parameters satisfy the constraints
                        check_constraints(res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],
                                          mu3)
                        if (tPrint):
                            print('Px   =',res.x[0])
                            print('pk   = ({}, {}, {})'.format(res.x[1],res.x[2],
                                                                 1 - res.x[1] - 
                                                                 res.x[2]))
                            print('mu   = ({}, {}, {})'.format(res.x[3],res.x[4],mu3))
                            print('SKL  = {:e}'.format(int(1.0 / res.fun)))
                            print('-'*60,'\n')
                        # Get final parameters from standard key length function
                        SKL,QBERx,phi_x,nX,nZ,mX,lambdaEC,sX0,sX1,vz1,sZ1,mpn = \
                            key_length(res.x)
                        # Store calculation parameters
                        fulldata[ci[2]*ndt + ci[3],:] = [ls+sysLoss,dt,int(1.0 / res.fun),QBERx,
                                             phi_x,nX,nZ,lambdaEC,sX0,sX1,vz1,sZ1,mpn,
                                             QBERI,Pec,Pap,NoPass,Rrate,eps_c,eps_s,
                                             res.x[0],res.x[1],res.x[2],1-res.x[1]
                                             -res.x[2],res.x[3],res.x[4],mu3,
                                             np.degrees(xi),min_elev,max_elev,
                                             shift_elev0]
                        # Store optimiser metrics
                        optdata[ci[2]*ndt + ci[3],:] = getOptData(Nopt,Ntot,x0,res,method)
                        # Store protocol parameters to initialise calculations
                        if np.isnan(int(1.0 / res.fun)) or np.isinf(int(1.0 / res.fun)):
                            x0i[:,ci[0],ci[1],ci[2],ci[3]] = x0_rand(mu3,xb,num_min)
                        else:
                            if int(1.0 / res.fun) > 0:
                                x0i[:,ci[0],ci[1],ci[2],ci[3]] = res.x
                            else:
                                x0i[:,ci[0],ci[1],ci[2],ci[3]] = x0_rand(mu3,xb,num_min)
                        count += 1 # Increment calculation counter
                        ci[3] += 1 # dt loop counter
                    ci[2] += 1 # ls loop counter
        else:
            # Calculate the SKL for the specified parameters (no optimisation).
            tc0 = perf_counter() # Start clock timer
            tp0 = process_time() # Start CPU timer
            
            ci[2] = 0
            for ls in range(ls_range[0],ls_range[1]+1,ls_range[2]):
                ci[3] = 0
                for dt in range(dt_range[0],dt_range[1]+1,dt_range[2]):
                    print('Calculation {}: Pec = {:5.2e}, QBERI = {:5.2e}, ls = {}, dt = {}'.format(
                            count+1,Pec,QBERI,ls,int(dt)))
                    # Calculate the secure key length for the specified parameters.
                    SKL,QBERx,phi_x,nX,nZ,mX,lambdaEC,sX0,sX1,vz1,sZ1,mpn = \
                        key_length(x0)
                    if (tPrint):
                        print('Px  =',x0[0])
                        print('pk  = ({}, {}, {})'.format(x0[1],x0[2],1 - x0[1] - 
                                                             x0[2]))
                        print('mu  = ({}, {}, {})'.format(x0[3],x0[4],mu3))
                        print('SKL = {:e}'.format(int(SKL)))
                        print('-'*60,'\n')
                    # Store calculation parameters
                    fulldata[ci[2]*ndt + ci[3],:] = [ls+sysLoss,dt,SKL,QBERx,phi_x,
                                         nX,nZ,lambdaEC,sX0,sX1,vz1,sZ1,mpn,QBERI,
                                         Pec,Pap,NoPass,Rrate,eps_c,eps_s,x0[0],x0[1],
                                         x0[2],1-x0[1]-x0[2],x0[3],x0[4],mu3,
                                         np.degrees(xi),min_elev,max_elev,shift_elev0]
                    count += 1 # Increment calculation counter
                    ci[3] += 1 # dt loop counter
                ci[2] += 1 # ls loop counter
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
        if (tWriteFiles):
            if tPrint:
                print('-'*60)
            if tFullData:
                # Write out full data in CSV format
                writeDataCSV(fulldata,outpath,outfile,header,'full loss & time data')
            if (tOptiData and ndt > 1):
                # Sort data by SKL per SysLoss
                sortdata = sort_data(fulldata,header,['SKL','SysLoss'],[True,False])
                # Only write out optimal SKL per SysLoss
                writeDataCSV(sortdata[::ndt,:],outpath,outfile+'_opt',header,
                             'optimal loss data')
            if (tOptimise and tMetrics):
                # Write optimiser metrics
                #print(res.keys()) # Prints available outputs for the object res
                writeDataCSV(optdata,outpath,outfile+'_metrics',opt_head,
                             'optimisation metrics')
            if tMultiOpt:
                if not (tOptiData and ndt > 1):
                    # Sort data by SKL per SysLoss
                    sortdata = sort_data(fulldata[:nls*ndt,:],header,['SKL','SysLoss'],[True,False])
                # Store optimal dt data per Pec, and QBERI
                cm0  = ci[0]*(nqi*nls) + ci[1]*(nls)
                cm1  = cm0 + nls
                multidata[cm0:cm1,:] = sortdata[::ndt,:]
            if tPrint:
                print('-'*60,'\n')
        ci[1] += 1 # QBERI loop counter
    ci[0] += 1 # Pec loop counter

if tMultiOpt:
    if tPrint:
        print('-'*60)
    # Filename for multi-optimal data
    multifile = outbase + '_multi-Pec-QBERI_{}GHz'.format(Rrate/1.0e9)
    # Write sorted data for all Pec and QBERIs
    writeDataCSV(multidata[:cm1,:],outpath,multifile,header,'multi-system optimal data')
    if tPrint:
        print('-'*60,'\n')
                
tc11 = perf_counter() # Stop clock timer
tp11 = process_time() # Stop CPU timer
tc   = tc11-tc00      # Calculation duration from clock
tp   = tp11-tp00      # Calculation duration from CPU
print('')
print('Final clock timer (s):',tc)
print('Final CPU timer (s):  ',tp,'\n')
print('All done!')
