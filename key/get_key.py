# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 14:54:00 2022

@author: Duncan McArthur
"""

from sys import float_info

import math
import numpy as np
from scipy.optimize import minimize

from key.protocols.init_efficient_BB84 import (x0_rand, x0_init, check_constraints)
from key.protocols.key_efficient_BB84 import (set_params, key_length, key_length_inv)
from channel.time_dependent_loss import get_losses
from output.outputs import (getOptData, writeDataCSV, write_data, writeMultiData,
                            get_timings, format_time)

__all__ = ['optimise_loop','check_opt','store_data','optimise_key',
           'arrange_output','SKL_opt_loop_loss_and_time',
           'SKL_loop_loss_and_time'<'SKL_sys_loop','sys_param_list',
           'args_fixed_list','SKL_main_loop']

###############################################################################

num_min = float_info.epsilon # A very small number

###############################################################################

def optimise_loop(tInit,x,x0i,ci,mu3,xb,args,bounds,cons,options,opt_params):
    """
    Execute a loop of optimisations of the main protocol parameters limited by
    either the total number of optimisations or the aggregate number of function
    evaluations. The initial parameters for optimisation are taken from
    previous successful (non-zero) optimisations, otherwise they are generated
    randomly.

    Parameters
    ----------
    tInit : bool
        Has an initial set of parameters been specified.
    x : float, array-like
        The initial set of parameters to use.
    x0i : float, array-like
        The final optimised parameters from previous calculations.
    ci : int, array-like
        Calculation loop counter array.
    mu3 : float
        Fixed protocol parameter. The intensity of the third pulse.
    xb : float, array-like
        The upper and lower bounds for the optimised parameters.
    args : dict
        Dictionary of extra optimiser arguments.
    bounds : obj
        Scipy bounds object containing optimised parameter bounds.
    cons : obj
        Scipy object containing the optimisation constraints.
    options : dict
        Dictionary of optimiser parameters.
    opt_params : dict
        Dictionary of parameters related to optimisation.

    Returns
    -------
    res : dict
        Dictionary of optimisation results.
    x0 : float, array-like
        Initial parameters used for final optimisation.
    Nopt : int
        Number of optimisations performed.
    Ntot : int
        Number of function evaluations.

    """
    # Re-set initial parameters (if required)
    Ninit = 0
    if tInit:
        # Store initial/fixed protocol parameters as an array
        x0 = x
    else:
        # Try to use the optimised parameters from the 
        # previous calculations as the initial values 
        # for this calculation.
        x0, Ninit = x0_init(x0i,ci,Ninit,mu3,xb,num_min)
        
    # Initial optimisation of SKL
    res = minimize(key_length_inv,x0,args=(args,),method=opt_params['method'],
                   jac=None,hess=None,hessp=None,bounds=bounds, 
                   constraints=cons,tol=None,callback=None, 
                   options=options)
    Ntot = res.nfev # Initilaise total No. of function evaluations
    Nopt = 1        # Number of optimisation calls
    # Re-run optimization until Nmax function evaluations
    # have been used. Take a copy of initial results to compare.
    x0_   = x0
    SKL_  = int(1.0 / res.fun)
    res_  = res
    Nzero = 0 # Number of times we get SKL == 0
    while Nopt < opt_params['NoptMin'] or Ntot < opt_params['Nmax']:
        # Initialise the optimised parameters
        x0, Ninit = x0_init(x0i,ci,Ninit,mu3,xb,num_min)
        # Calculate optimised SKL
        res = minimize(key_length_inv,x0,args=(args,),method=opt_params['method'],
                       jac=None,hess=None,hessp=None,bounds=bounds, 
                       constraints=cons,tol=None,callback=None, 
                       options=options)
        Nopt += 1 # Increment optimisation counter
        if int(1.0 / res.fun) > 0:
            if int(1.0 / res.fun) > SKL_:
                if Nopt >= opt_params['NoptMin'] and opt_params['tStopBetter']:
                    break # A better value was found!
                else:
                    # Store new set of best parameters
                    x0_  = x0
                    res_ = res
                    SKL_ = int(1.0 / res.fun)
            else:
                # Reset to the best parameters
                x0  = x0_
                res = res_
        else:
            # SKL = 0. Reset to the 'best' parameters,
            # (may still give SKL = 0).
            Nzero += 1
            if Nopt > opt_params['NoptMin']:
                if Nzero / (Nopt - 1) == 1:
                    # We get SKL = 0 every time.
                    if opt_params['tStopZero']:
                        break
                    x0  = x0_
                    res = res_
        Ntot += res.nfev

    return res, x0, Nopt, Ntot

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def check_opt(res,mu3,method,tPrint):
    """
    Check optimiser output is within bounds and constraints of protocol.

    Parameters
    ----------
    res : dict
        Dictionary of optimisation results.
    mu3 : float
        Intensity of pulse three (vacuum).
    method : str
        Optimisation method.
    tPrint : bool
        Print output to std out?

    Returns
    -------
    None.

    """
    #print('Nopt =',Nopt)
    #print('Ntot =',Ntot)
    if (tPrint):
        if (res.success):
            if method in ['trust-constr','SLSQP']:
                print('Nit  =', res.nit) # Number of iterations
            else:
                print('Nit  =', res.nfev) # Number of iterations
        else:
            print("Optimiser status = {0}: {1}".format(res.status,res.message))
    # Check if optimised parameters satisfy the constraints
    check_constraints(res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],mu3)
    if (tPrint):
        print('Px   =',res.x[0])
        print('pk   = ({0}, {1}, {2})'.format(res.x[1],res.x[2],1 - res.x[1] - 
                                              res.x[2]))
        print('mu   = ({0}, {1}, {2})'.format(res.x[3],res.x[4],mu3))
        print('SKL  = {0:e}'.format(int(1.0 / res.fun)))
        print('-'*80,'\n')
    return None

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def store_data(res,Nopt,Ntot,x0,method,mu3,xb,args):
    """
    Store results, calculation parameters and optimisation metrics as arrays.

    Parameters
    ----------
    res : dict
        Dictionary of optimisation results.
    Nopt : int
        Number of optimisations performed.
    Ntot : int
        Number of function evaluations.
    x0 : float, array-like
        Initial parameters used for final optimisation.
    method : str
        Optimisation method.
    mu3 : float
        Intensity of pulse three (vacuum).
    xb : float, array-like
        The upper and lower bounds for the optimised parameters.
    args : dict
        Dictionary of extra optimiser arguments.

    Returns
    -------
    fulldata : float, array-like
        Calculation output parameters.
    optdata : list
        Optimisation metrics and parameters.
    x0i : float, array-like
        The final optimised protocol parameters including this calculation.

    """
    # Get final parameters from standard key length function
    SKL,QBERx,phi_x,nX,nZ,mX,lambdaEC,sX0,sX1,vz1,sZ1,mpn = key_length(res.x,args)
    # Store calculation parameters
    fulldata = [SKL,QBERx,phi_x,nX,nZ,mX,lambdaEC,sX0,sX1,vz1,sZ1,mpn]
    # Store optimiser metrics
    optdata = getOptData(Nopt,Ntot,x0,res,method)
    # Store protocol parameters to initialise calculations
    if np.isnan(int(1.0 / res.fun)) or np.isinf(int(1.0 / res.fun)):
        # SKL = 0 or was invalid. Store a random set of optimised params for 
        # intialising future calculations
        x0i = x0_rand(mu3,xb,num_min)
    else:
        if int(1.0 / res.fun) > 0:
            # SKL > 0. Store optimised parameters to intialise future 
            # calculations
            x0i = res.x
        else:
            # Unexpected SKL result. Store a random set of optimised params for 
            # intialising future calculations
            x0i = x0_rand(mu3,xb,num_min)
    return fulldata, optdata, x0i

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def optimise_key(tInit,x,x0i,ci,mu3,xb,args,bounds,cons,options,opt_params,
                 tPrint):
    """
    Find the optimised parameters, by optimising repeteadly, then check the
    parameters and then store the output data in arrays.

    Parameters
    ----------
    tInit : bool
        Have initial values for the protocol parameters been specified?
    x : float, array-like
        The initial set of parameters to use.
    x0i : float, array-like
        The final optimised protocol parameters from previous calculations.
    ci : int, array-like
        Calculation loop counter array.
    mu3 : float
        Fixed protocol parameter. The intensity of the third pulse
    xb : float, array-like
        The upper and lower bounds for the optimised parameters.
    args : dict
        Dictionary of extra optimiser arguments.
    bounds : obj
        Scipy bounds object containing optimised parameter bounds.
    cons : obj
        Scipy object containing the optimisation constraints.
    options : dict
        Dictionary of optimiser parameters.
    opt_params : dict
        Dictionary of parameters related to optimisation.
    tPrint : bool
        Print output to std out?

    Returns
    -------
    res : dict
        Dictionary of optimisation results.
    fulldata : float, array-like
        Calculation output parameters.
    optdata : list
        Optimisation metrics and parameters.
    x0i : float, array-like
        The final optimised protocol parameters including this calculation.

    """
    # Run optimisation loop
    res, x0, Nopt, Ntot = optimise_loop(tInit,x,x0i,ci,mu3,xb,args,bounds,cons,
                                        options,opt_params)
    # Check optimised output
    check_opt(res,mu3,opt_params['method'],tPrint)
    # Retrive data, metrics, and initial values
    fulldata, optdata, x0i = store_data(res,Nopt,Ntot,x0,opt_params['method'],
                                        mu3,xb,args)
    return res, fulldata, optdata, x0i

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def arrange_output(SysLoss,ls,dt,mu3,QBERI,Pec,theta_max,x,SKLdata,sys_params):
    """
    Put calculation output parameters into an ordered list.

    Parameters
    ----------
    sysLoss : float
        The nominal loss or system loss metric (dB). For symmetric transmission
        windows this is the system loss at zenith.
    ls : float
        Excess system loss (dB).
    dt : int
        Transmission window half-width (s).
    mu3 : float
        Fixed protocol parameter. The intensity of the third pulse.
    QBERI : float
        Intrinsic quantum bit error rate.
    Pec : float
        Extraneous count rate/probability.
    theta_max : float
        Maximum elevation of satellite overpass (rad).
    x : float, array-like
        The initial set of parameters to use.
    SKLdata : float, array-like
        Calculation output parameters.
    sys_params : dict
        Additional system parameters to be included in fulldata.

    Returns
    -------
    list
        Ordered list of data to write out.

    """
    # [dt,ls,QBERI,Pec,theta_max,SKL,QBERx,...]
    return [dt,ls,QBERI,Pec,sys_params[0],*SKLdata,x[0],*x[:3],1-x[1]-x[2],
            *x[3:],mu3,*sys_params[1:],SysLoss]

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def SKL_opt_loop_loss_and_time(count,ci,ni,theta_max,Pec,QBERI,ls_range,
                               dt_range,tInit,x,x0i,mu3,xb,args_fixed,bounds,
                               cons,options,opt_params,tPrint,fulldata,
                               optdata,sys_params,sysLoss):
    """
    Perfom secret key optimisations for iterated values of the transmission
    window half-width (dt) and the excess system loss (ls).

    Parameters
    ----------
    count : int
        Absolute calculation counter.
    ci : int, array-like
        Calculation loop counter.
    ni : int, array-like
        The number of iterations in each loop.
    theta_max : float
        Maximum elevation of satellite overpass (rad).
    Pec : float
        Extraneous count rate/probability.
    QBERI : float
        Intrinsic quantum bit error rate.
    ls_range : int, array-like (3)
        Start, stop, and No. of step values for the excess loss loop.
    dt_range : int, array-like (3)
        Start, stop, and step values for the transmission window (half-width) 
        loop (s).
    tInit : bool
        Have initial values for the protocol parameters been specified?
    x : float, array-like
        The initial set of parameters to use.
    x0i : float, array-like
        The initial parameters to use.
    ci : int, array-like
        Calculation loop counter array.
    mu3 : float
        Fixed protocol parameter. The intensity of the third pulse.
    xb : float, array-like
        The upper and lower bounds for the optimised parameters.
    args_fixed : dict
        Dictionary of arguments required by key_length functions.
    bounds : obj
        Scipy bounds object containing optimised parameter bounds.
    cons : obj
        Scipy object containing the optimisation constraints.
    options : dict
        Dictionary of optimiser parameters.
    opt_params : dict
        Dictionary of parameters related to optimisation.
    tPrint : bool
        Print output to std out?
    fulldata : float, array-like
        Calculation output parameters.
    optdata : list
        Optimisation metrics and parameters.
    sys_params : dict
        Additional system parameters to be included in fulldata.
    sysLoss : float
        The nominal loss or system loss metric (dB). For symmetric transmission
        windows this is the system loss at zenith.

    Returns
    -------
    fulldata : float, array-like
        Calculation output parameters.
    optdata : list
        Optimisation metrics and parameters.
    x0i : float, array-like
        The final optimised protocol parameters including this calculation.
    count : int
        Updated absolute calculation counter.

    """
    # Perform the optimization of the secure key length
    ci[3] = 0
    for ls in np.linspace(ls_range[0],ls_range[1],ls_range[2]):
        ci[4] = 0
        for dt in range(dt_range[0],dt_range[1]+1,dt_range[2]):
            if tPrint:
                print('Calculation {}: tm ({}/{}), Pec ({}/{}), QBERI ({}/{}), ls ({}/{}), dt ({}/{})'.format(
                                        count+1,ci[0]+1,ni[0],ci[1]+1,ni[1],ci[2]+1,ni[2],ci[3]+1,ni[3],ci[4]+1,ni[4]))
                print('theta_max = {:5.2f}, Pec = {:5.2e}, QBERI = {:5.2e}, ls = {}, dt = {}'.format(
                    np.degrees(theta_max),Pec,QBERI,ls,int(dt)))
            # Store key params in a dict
            args = set_params(Pec,QBERI,ls,dt,*args_fixed)
            # Optimise key length
            res, SKLdata, optdata[ci[3]*ni[4] + ci[4],:], x0i[:,ci[0],ci[1],ci[2],ci[3],ci[4]] = \
                optimise_key(tInit,x,x0i,ci,mu3,xb,args,bounds,cons,options,
                             opt_params,tPrint)
            # Store output data
            fulldata[ci[3]*ni[4] + ci[4],:] = arrange_output(sysLoss+ls,ls,dt,
                                                             mu3,QBERI,Pec,
                                                             theta_max,res.x,
                                                             SKLdata,
                                                             sys_params)
            count += 1 # Increment calculation counter
            ci[4] += 1 # dt loop counter
        ci[3] += 1 # ls loop counter

    return fulldata, optdata, x0i, count

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def SKL_loop_loss_and_time(count,ci,ni,theta_max,Pec,QBERI,ls_range,dt_range,
                           x0,mu3,args_fixed,tPrint,fulldata,sys_params,
                           sysLoss):
    """
    Perfom secret key (non-optimised) calculations for iterated values of the 
    transmission window half-width (dt) and the excess system loss (ls).

    Parameters
    ----------
    count : int
        Absolute calculation counter.
    ci : int, array-like
        Calculation loop counter.
    ni : int, array-like
        The number of iterations in each loop.
    theta_max : float
        Maximum elevation of satellite overpass (rad).
    Pec : float
        Extraneous count rate/probability.
    QBERI : float
        Intrinsic quantum bit error rate.
    ls_range : int, array-like (3)
        Start, stop, and No. of step values for the excess loss loop.
    dt_range : int, array-like (3)
        Start, stop, and step values for the transmission window (half-width) 
        loop (s).
    x0 : float, array-like
        The set of protocol parameters to use.
    mu3 : float
        Fixed protocol parameter. The intensity of the third pulse.
    args_fixed : dict
        Dictionary of arguments required by key_length functions.
    tPrint : bool
        Print output to std out?
    fulldata : float, array-like
        Calculation output parameters.
    sys_params : dict
        Additional system parameters to be included in fulldata.
    sysLoss : float
        The nominal loss or system loss metric (dB). For symmetric transmission
        windows this is the system loss at zenith.

    Returns
    -------
    fulldata : float, array-like
        Calculation output parameters.
    count : int
        Updated absolute calculation counter.

    """
    
    # Perform the optimization of the secure key length
    ci[3] = 0
    for ls in np.linspace(ls_range[0],ls_range[1],ls_range[2]):
        ci[4] = 0
        for dt in range(dt_range[0],dt_range[1]+1,dt_range[2]):
            if tPrint:
                print('Calculation {}: tm ({}/{}), Pec ({}/{}), QBERI ({}/{}), ls ({}/{}), dt ({}/{})'.format(
                                        count+1,ci[0]+1,ni[0],ci[1]+1,ni[1],ci[2]+1,ni[2],ci[3]+1,ni[3],ci[4]+1,ni[4]))
                print('theta_max = {:5.2f}, Pec = {:5.2e}, QBERI = {:5.2e}, ls = {}, dt = {}'.format(
                    np.degrees(theta_max),Pec,QBERI,ls,int(dt)))
            # Store key params in a dict
            args = set_params(Pec,QBERI,ls,dt,*args_fixed)
            # SKL calculation
            SKL,QBERx,phi_x,nX,nZ,mX,lambdaEC,sX0,sX1,vz1,sZ1,mpn = key_length(x0,args)
            # Make a list of output data
            SKLdata = [SKL,QBERx,phi_x,nX,nZ,mX,lambdaEC,sX0,sX1,vz1,sZ1,mpn]
            # Store output data
            fulldata[ci[3]*ni[4] + ci[4],:] = arrange_output(sysLoss+ls,ls,dt,
                                                             mu3,QBERI,Pec,
                                                             theta_max,x0,
                                                             SKLdata,
                                                             sys_params)
            count += 1 # Increment calculation counter
            ci[4] += 1 # dt loop counter
        ci[3] += 1 # ls loop counter
    return fulldata, count

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def SKL_sys_loop(count,ci,ni,x,x0i,xb,theta_max,dt_range,main_params,opt_params,
                 args_fixed,bounds,cons,options,header,opt_head,fulldata,optdata,
                 multidata,sys_params,sysLoss):
    """
    Calculate the SKL over the main iterated parameter loops. 

    Parameters
    ----------
    count : int
        Absolute calculation counter.
    ci : int, array-like
        Calculation loop counter.
    ni : int, array-like
        The number of iterations in each loop.
    x : float, array-like
        The initial set of protocol parameters to use.
    x0i : float, array-like
        The final protocol parameters from previous calculations.
    xb : float, array-like
        The upper and lower bounds for the optimised parameters.
    theta_max : float
        Maximum elevation of satellite overpass (rad).
    dt_range : int, array-like (3)
        Start, stop, and step values for the transmission window (half-width) 
    main_params : dict
        Dictionary of main calculation parameters.
    opt_params : dict
        Dictionary of parameters related to optimisation.
    args_fixed : dict
        Dictionary of arguments required by key_length functions.
    bounds : obj
        Scipy bounds object containing optimised parameter bounds.
    cons : obj
        Scipy object containing the optimisation constraints.
    options : dict
        Dictionary of optimiser parameters.
    header : str
        Comma separated string of output parameters corresponding to columns
        in fulldata. Used as header when writing output to file.
    opt_head : str
        Comma separated string of optimiser metrics and parameters corresponding
        to columns of optdata. Used as a header when writing output to file.
    fulldata : float, array-like
        Calculation output parameters.
    optdata : list
        Optimisation metrics and parameters.
    multidata : float, array-like
        Optimal dt output parameters for all calculations.
    sys_params : dict
        Additional system parameters to be included in fulldata.
    sysLoss : float
        The nominal loss or system loss metric (dB). For symmetric transmission
        windows this is the system loss at zenith.

    Returns
    -------
    multidata : float, array-like
        Updated optimal dt output parameters for all calculations.
    count : int
        Updated absolute calculation counter.

    """
    # Extract parameters for the calculation
    tOptimise = main_params['opt']['tOptimise'] # Perform an optimised calculation?
    tInit     = main_params['opt']['tInit']     # Use specified intial protocol paraneters for optimised calculation?
    mu3       = main_params['fixed']['mu3']     # The intensity of pulse 3 (BB84 decoy states).
    ls_range  = main_params['iter']['ls']       # Start, stop and step values for the excess loss loop
    
    ci[1] = 0  
    for Pec in main_params['iter']['Pec']:
        ci[2] = 0
        for QBERI in main_params['iter']['QBERI']:
            #print('System: '.format())
            outfile = main_params['out']['out_base'] + \
                '_th_m_{:5.2f}_Pec_{}_QBERI_{}'.format(np.degrees(theta_max),
                                                       Pec,QBERI)
            
            # Start clock and CPU timer
            tc0, tp0 = get_timings()

            if tOptimise:
                # Find the optimised SKL and protocol parameters
                fulldata, optdata, x0i, count = \
                    SKL_opt_loop_loss_and_time(count,ci,ni,theta_max,Pec,
                                               QBERI,ls_range,dt_range,tInit,x,
                                               x0i,mu3,xb,args_fixed,bounds,
                                               cons,options,opt_params,
                                               main_params['out']['tPrint'],
                                               fulldata,optdata,sys_params,
                                               sysLoss)
            else:
                # Calculate the SKL for a given set of protocol parameters
                fulldata, count = \
                    SKL_loop_loss_and_time(count,ci,ni,theta_max,Pec,QBERI,
                                           ls_range,dt_range,x,mu3,args_fixed,
                                           main_params['out']['tPrint'],
                                           fulldata,sys_params,sysLoss)

            # Stop clock and CPU timer
            tc1, tp1 = get_timings()
            
            #*******************************************************************
            # Print the calculation timings
            #*******************************************************************
            tc = tc1-tc0 # Calculation duration from clock
            tp = tp1-tp0 # Calculation duration from CPU
            if main_params['out']['tPrint']:
                print('Clock timer:',format_time(tc))
                print('CPU timer:  ',format_time(tp),'\n')
            
            #*******************************************************************
            #*******************************************************************
            # Sort and output data
            #*******************************************************************
            #*******************************************************************
            if (main_params['out']['tWriteFiles']):
                multidata = write_data(main_params['out'],tOptimise,ni,ci,
                                       header,opt_head,outfile,fulldata,
                                       multidata,optdata)
            
            ci[2] += 1 # QBERI loop counter
        ci[1] += 1 # Pec loop counter
    return multidata, count
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def sys_param_list(fixed_params,max_elev):
    """
    Generate a list of fixed system parameters.

    Parameters
    ----------
    fixed_params : dict
        Dictionary of fixed arguments required by key_length functions.
    max_elev : float
        Maximum elevation of the staellite overpass (rad).

    Returns
    -------
    sys_params : dict
        Dictionary of fixed system parameters.

    """
    sys_params = []
    sys_params.append(max_elev)                # The maximum elevation of the satellite overpass
    sys_params.append(fixed_params['eps_c'])   # The correctness parameter
    sys_params.append(fixed_params['eps_s'])   # The secrecy parameter
    sys_params.append(fixed_params['Pap'])     # Probability of an afterpulse event
    sys_params.append(fixed_params['NoPass'])  # Number of satellite overpasses
    sys_params.append(fixed_params['Rrate'])   # Source repetition rate
    sys_params.append(fixed_params['minElev']) # The minimum satellite elevation used for transmission
    sys_params.append(fixed_params['shift0'])  # Number of time steps to shift the t=0 point from zenith
    return sys_params

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def args_fixed_list(fixed_params,calc_params,FSeff,time0pos):
    """
    Extract fixed parameters from a dictionary and return as a list.

    Parameters
    ----------
    fixed_params : dict
        Dictionary of fixed system parameters.
    calc_params : dict
        Dictionary of calculation parameters.
    FSeff : float, array-like
        The free space transmission losses (as efficiency) as a function of time.
    time0pos : int
        Index of the t=0 point of the transmission window in the FSeff array.

    Returns
    -------
    list
        Fixed system parameters.

    """
    mu3         = fixed_params['mu3']     # Intensity of pulse 3 (vacuum)
    Pap         = fixed_params['Pap']     # Probability of an afterpulse event
    Npulse      = fixed_params['Npulse']  # The number of transmitted pulses
    eps_c       = fixed_params['eps_c']   # The correctness parameter
    eps_s       = fixed_params['eps_s']   # The secrecy parameter
    NoPass      = fixed_params['NoPass']  # Number of satellite overpasses
    boundFunc   = calc_params['bounds']   # Type of tail bounds
    errcorrFunc = calc_params['funcEC']   # Error correction function
    fEC         = calc_params['fEC']      # Error correction factor   
    num_zero    = calc_params['num_zero'] # Small value used to approximate zero
    return [mu3,time0pos,Pap,FSeff,Npulse,boundFunc,eps_c,eps_s,num_zero,
            errcorrFunc,fEC,NoPass]

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def SKL_main_loop(main_params,adv_params,x,x0i,xb,ci,ni,f_atm,bounds,cons,
                  options,header,opt_head,fulldata,optdata,multidata):
    """
    Calculate the SKL using either optimised or specified protocol parameters
    for iterated values of:
        theta_max : Maximum elevation angle (rad)
        Pec : Probability of extraneous counts
        QBERI : Intrinsic quantum bit error rate
        ls : Excess system loss (dB)
        dt: Transmission window half-width (s)

    Parameters
    ----------
    main_params : dict
        Dictionary of general calculation parameters.
    adv_params : dict
        Dictionary of advanced calculation parameters.
    x : float, array-like
        The initial set of protocol parameters to use.
    x0i : float, array-like
        The final protocol parameters from previous calculations.
    xb : float, array-like
        The upper and lower bounds for the optimised parameters.
    ci : int, array-like
        Calculation loop counter.
    ni : int, array-like
        The number of iterations in each loop.
    f_atm : function
        Atmospheric transmissivity vs elevation angle (rad) function.
    bounds : obj
        Scipy bounds object containing optimised parameter bounds.
    cons : obj
        Scipy object containing the optimisation constraints.
    options : dict
        Dictionary of optimiser parameters.
    header : str
        Comma separated string of output parameters corresponding to columns
        in fulldata. Used as header when writing output to file.
    opt_head : str
        Comma separated string of optimiser metrics and parameters corresponding
        to columns of optdata. Used as a header when writing output to file.
    fulldata : float, array-like
        Calculation output parameters.
    optdata : list
        Optimisation metrics and parameters.
    multidata : float, array-like
        Optimal dt output parameters for all calculations.

    Returns
    -------
    None.

    """

    count = 0 # Initialise calculation counter
    ci[0] = 0
    for theta_max in main_params['iter']['theta_max']:
        # Retrieve loss data array
        loss_data = get_losses(theta_max,main_params['loss'],f_atm,
                               main_params['out']['tPrint'],
                               main_params['out']['out_path'])

        # Free space loss in dB (to be converted to efficiency)
        # Returns the (t,) array FSeff, where t is the total number of time-slots
        FSeff = loss_data[:,2]

        # Find the time slot at the centre of the pass where t = 0.
        time0pos   = np.where(loss_data[:,0] == 0)[0][0]
        time0elev  = loss_data[time0pos,1] # Elevation angle at t = 0 (rads).
        time0shift = time0pos # Take a temporary copy to use later.

        # Nominal system loss: based on zenith coupling efficiency and nominal losses
        # if xi == 0.0:
        #     sysLoss = -10*(math.log10(FSeff[time0pos]) + math.log10(eta))
        # else:
        #     slm     = eta_loss_metric(hsat,h0,wvl,aT,aR,w0,f_atm,eta_int)
        #     sysLoss = slm - 10*math.log10(eta)
        sysLoss = -10*(math.log10(FSeff[time0pos]) + 
                       math.log10(main_params['fixed']['eta']))

        # Maximum elevation angle (degs) of satellite pass
        #max_elev = np.degrees(cvs[time0pos,1])
        max_elev = np.degrees(time0elev)
    
        if (main_params['fixed']['shift0'] != 0.0):
            # Shift the elevation angle taken as t = 0 away from elev = 90 deg.
            # Find the first array index for an elevation angle greater than, or equal
            # to, the shifted angle requested.
            time0pos   = np.where(loss_data[:,1] >= (time0elev - 
                                                     np.radians(main_params['fixed']['shift0'])))[0][0]
            time0elev  = loss_data[time0pos,1] # New elevation angle at t = 0 (rads).
            time0shift = abs(time0pos - time0shift) # Shift in time slots between old and new t = 0.
        else:
            # No shift requested, t = 0 is at elev = 90 deg.
            time0shift = 0 # Reset this value to zero.

        min_elev   = main_params['fixed']['minElev'] # Reset value
        # Find first dt value corresponding to an elevation greater than min_elev
        minElevpos = np.where(loss_data[:,1] >= np.radians(min_elev))[0][0] # Index of first value
        dt_elev    = loss_data[minElevpos,0] # Max value of dt less than, or equal to, the
                                             # minimum elevation angle

        # Check dt_range is within bounds
        dt_max = int(0.5*(len(FSeff) - 1) - time0shift) # Maximum time window half-width
        dt_range                     = np.asarray(main_params['iter']['dt']) # Reset initial loop values
        dt_range[dt_range < 0]       = 0         # All values must be positive
        dt_range[dt_range > dt_max]  = dt_max    # Max cannot exceed No. of time-slots
        dt_range[dt_range > dt_elev] = dt_elev   # Limit range by minimum elevation value
    
        # The time range changes with xi
        if (dt_range[0] == dt_range[1] or dt_range[2] == 0):
            ni[4] = 1
        else:
            ni[4] = int((dt_range[1] - dt_range[0]) / float(dt_range[2])) + 1

        # Get minimum elevation for transmission (possibly greater than value specified)
        minElevpos = np.where(loss_data[:,0] <= dt_range[1])[0][0] # Index of first value
        min_elev   = np.degrees(loss_data[minElevpos,1]) # Minimum elevation (degs)
        # Store list of fixed params for output data file
        sys_params = sys_param_list(main_params['fixed'],max_elev)
        # Store list of optimiser arguments
        args_fixed = args_fixed_list(main_params['fixed'],adv_params['calc'],
                                     FSeff,time0pos)
        
        # Run main SKL loops
        multidata, count = \
            SKL_sys_loop(count,ci,ni,x,x0i,xb,theta_max,dt_range,main_params,
                         adv_params['opt'],args_fixed,bounds,cons,options,
                         header,opt_head,fulldata,optdata,multidata,
                         sys_params,sysLoss)
            
        ci[0] += 1 # theta_max loop counter

    if main_params['out']['tdtOptData']:
        # Write out dt optimised data over all other parameters
        writeMultiData(main_params['out'],header,multidata)
        if main_params['out']['tPrint']:
            print('-'*80)
