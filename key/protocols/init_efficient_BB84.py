# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 01:53:37 2021

@author: Duncan McArthur
"""

import numpy as np

__all__ = ['get_x_bounds','x0_rand','x0_init','check_constraints',
           'bool_constraints']

###############################################################################

def get_x_bounds(opt_dict,mu3,num_min):
    """
    Returns the intial values and upper & lower bounds for the optimised 
    parameters.

    Parameters
    ----------
    opt_dict : dict
        Dictionary of parameters related to optimisation.
    mu3 : float
        Intensity of pulse 3 (vacuum).
    num_min : float
        An arbitrarily small number.

    Returns
    -------
    x : float, array
        Optimised parameters initial values.
    xb : float, array
        Optimised parameters upper & lower bounds.

    """
    xb = np.zeros((5,2))
    xb[0,:] = opt_dict['Px'][3:None]
    xb[1,:] = opt_dict['P1'][3:None]
    xb[2,:] = opt_dict['P2'][3:None]
    xb[3,:] = opt_dict['mu1'][3:None]
    xb[4,:] = opt_dict['mu2'][3:None]

    x = np.zeros((5,))
    if opt_dict['Px'][1]:
        # Initial value for Px is specified
        x[0] = opt_dict['Px'][2]
    else:
        # Initial value for Px is randomly allocated
        x[0] = np.random.rand() * (xb[0,1] - xb[0,0] - 2*num_min) + xb[0,0] + \
                num_min
    if opt_dict['P1'][1] and opt_dict['P2'][1]:
        # Initial values for both P1 and P2 are specified
        x[1] = opt_dict['P1'][2]
        x[2] = opt_dict['P2'][2]
    elif opt_dict['P1'][1]:
        # Initial value for P1 is specified, P2 is randomly allocated
        x[1] = opt_dict['P1'][2]
        x[2] = 1.0
        while (x[1] + x[2] >= 1.0):
            x[2] = np.random.rand() * (min(xb[2,1],1 - x[1]) - xb[2,0] - \
                                        2*num_min) + xb[2,0] + num_min
    elif opt_dict['P2'][1]:
        # Initial value for P2 is specified, P1 is randomly allocated
        x[1] = 1.0
        x[2] = opt_dict['P2'][2]
        while (x[1] + x[2] >= 1.0):
            x[1] = np.random.rand() * (min(xb[1,1],1 - x[2]) - xb[1,0] - \
                                        2*num_min) + xb[1,0] + num_min
    else:
        # Initial values for P1 and P2 are randomly allocated
        x[1], x[2] = 1.0, 1.0
        while (x[1] + x[2] >= 1.0):
            x[1] = np.random.rand() * (xb[1,1] - xb[1,0] - 2*num_min) + \
                                        xb[1,0] + num_min
            x[2] = np.random.rand() * (min(xb[2,1],1 - x[1]) - xb[2,0] - \
                                        2*num_min) + xb[2,0] + num_min
    if opt_dict['mu1'][1]:
        # Initial value for mu1 is specified
        x[3] = opt_dict['mu1'][2]
    else:
        # Initial value for mu1 is randomly allocated
        x[3] = np.random.rand() * (xb[3,1] - max(xb[3,0],2*mu3) - 2*num_min) + \
                                    max(xb[3,0],2*mu3) + num_min
    if opt_dict['mu2'][1]:
        # Initial value for mu2 is specified
        x[4] = opt_dict['mu2'][2]
    else:
        # Initial value for mu2 is randomly allocated
        x[4] = np.random.rand() * (min(xb[4,1],x[3]) - max(xb[4,0],mu3) - \
                                   2*num_min) + max(xb[4,0],mu3) + num_min
    return x, xb

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
    while (pk1_i + pk2_i >= 1.0):
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

def x0_init(x0i,ci,Ninit,mu3,xb,num_min):
    """
    Initialise the optimised protocol parameters.
    
    First try to use parameters from previous calculations, otherwise randomly
    intialise the parameters.
    
    From previous calculations prioritise values from: dt > ls > xi > QBERI > Pec.

    Parameters
    ----------
    x0i : float, array
        Array of final parameters from previous calculations.
    ci : int, array
        Loop counters.
    Ninit : int
        Initialisation counter.
    mu3 : float
        Intensity of pulse 3 (vacuum).
    xb : float, array-like
        Upper and lower bounds for the protocol parameters. (5,2)
    num_min : float
        An arbitrarily small number.

    Returns
    -------
    float, array
        Inital values for optimised parameters.

    """

    if Ninit < 5:
        # Select from the 5 intialisation options (otherwise initialise randomly)
        if Ninit == 0:
            if ci[4] > 0:
                # Use parameters from previous dt
                return x0i[:,ci[0],ci[1],ci[2],ci[3],ci[4]-1], Ninit + 1
            elif ci[3] > 0:
                # Use parameters from the previous ls
                return x0i[:,ci[0],ci[1],ci[2],ci[3]-1,ci[4]], Ninit + 2
            elif ci[0] > 0:
                # Use parameters from the previous xi
                return x0i[:,ci[0]-1,ci[1],ci[2],ci[3],ci[4]], Ninit + 3
            elif ci[2] > 0:
                # Use parameters from the previous QBERI
                return x0i[:,ci[0],ci[1],ci[2]-1,ci[3],ci[4]], Ninit + 4
            elif ci[1] > 0:
                # Use parameters from the previous Pec
                return x0i[:,ci[0],ci[1]-1,ci[2],ci[3],ci[4]], Ninit + 5
            else:
                Ninit = 4
        elif Ninit == 1:
            if ci[3] > 0:
                # Use parameters from the previous ls
                return x0i[:,ci[0],ci[1],ci[2],ci[3]-1,ci[4]], Ninit + 1
            elif ci[0] > 0:
                # Use parameters from the previous xi
                return x0i[:,ci[0]-1,ci[1],ci[2],ci[3],ci[4]], Ninit + 2
            elif ci[2] > 0:
                # Use parameters from the previous QBERI
                return x0i[:,ci[0],ci[1],ci[2]-1,ci[3],ci[4]], Ninit + 3
            elif ci[1] > 0:
                # Use parameters from the previous Pec
                return x0i[:,ci[0],ci[1]-1,ci[2],ci[3],ci[4]], Ninit + 4
            else:
                Ninit = 4
        elif Ninit == 2:
            if ci[0] > 0:
                # Use parameters from the previous xi
                return x0i[:,ci[0]-1,ci[1],ci[2],ci[3],ci[4]], Ninit + 1
            elif ci[2] > 0:
                # Use parameters from the previous QBERI
                return x0i[:,ci[0],ci[1],ci[2]-1,ci[3],ci[4]], Ninit + 2
            elif ci[1] > 0:
                # Use parameters from the previous Pec
                return x0i[:,ci[0],ci[1]-1,ci[2],ci[3],ci[4]], Ninit + 3
            else:
                Ninit = 4
        elif Ninit == 3:
            if ci[2] > 0:
                # Use parameters from the previous QBERI
                return x0i[:,ci[0],ci[1],ci[2]-1,ci[3],ci[4]], Ninit + 1
            elif ci[1] > 0:
                # Use parameters from the previous Pec
                return x0i[:,ci[0],ci[1]-1,ci[2],ci[3],ci[4]], Ninit + 2
            else:
                Ninit = 4
        elif Ninit == 4:
            if ci[1] > 0:
                # Use parameters from the previous Pec
                return x0i[:,ci[0],ci[1]-1,ci[2],ci[3],ci[4]], Ninit + 1

    return x0_rand(mu3,xb,num_min), Ninit + 1

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