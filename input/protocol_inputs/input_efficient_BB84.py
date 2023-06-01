# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 12:17:17 2021

@author: Duncan McArthur
"""

import numpy as np

from ..parse_input_string import (str2bool, strip_quote_input, str_is_None)
from ..convert_input_string import (list_str_to_float, list_str_to_int, 
                                    read_protocol_param)
from ..optimiser_params import (convert_str_params_opt, default_params_opt, 
                                check_params_opt)
from ..output_params import convert_str_params_out

__all__ = ['convert_str_params','convert_str_params_adv',
           'convert_str_params_adv','check_params']

###############################################################################

# List of supported tail bounds for this protocol
boundOpts   = ['Chernoff','Hoeffding','Asymptotic']
# List of supprted error correction methods for this protocol
errcorrOpts = ['logM','block','mXtot','None']

###############################################################################

def convert_str_params(param_str):
    """
    Converts specific protocol and general calculation parameters from string 
    to values and returns as a dictionary.

    Parameters
    ----------
    param_str : string, list
        List of string parameters read from file.

    Raises
    ------
    ValueError
        If the number of string parameters is less than the minimum expected.

    Returns
    -------
    params : mixed, dictionary
        Protocol and calcualtion parameter values.

    """
    # nMinParams = 30
    # nParams    = len(param_str) # Total number of lines in the input file
    # if nParams < nMinParams:
    #     raise ValueError('nParams = {} < nMinParams = {}'.format(nParams,nMinParams))
    count = 0
    ###########################################################################
    # Optimisable parameters: returns [tOptimise, tInit, val, lb, ub]
    ###########################################################################
    opt_params = {}
    # Optimise all protocol parameters, else specify
    opt_params['tOptimise'] = str2bool(param_str[count])
    #print('tOptimise = ',opt_params['tOptimise'])
    count += 1
    
    # Probability Alice sends an X basis signal
    opt_params['Px'] = read_protocol_param(param_str[count],opt_params['tOptimise'])
    count += 1
    # Probability Bob measures an X basis signal
    # opt_params['PxB'] = read_protocol_param(param_str[count],opt_params['tOptimise'])
    # count += 1
    # Probabiltity Alice prepares a state with intensity 1
    opt_params['P1']  = read_protocol_param(param_str[count],opt_params['tOptimise'])
    count += 1
    # Probabiltity Alice prepares a state with intensity 2
    opt_params['P2']  = read_protocol_param(param_str[count],opt_params['tOptimise'])
    count += 1
    # Intensity of state 1
    opt_params['mu1'] = read_protocol_param(param_str[count],opt_params['tOptimise'])
    count += 1
    # Intensity of state 2
    opt_params['mu2'] = read_protocol_param(param_str[count],opt_params['tOptimise'])
    count += 1
    
    ###########################################################################
    # Fixed parameters
    ###########################################################################
    fixed_params = {}
    # Intensity of state 3 (vacuum)
    fixed_params['mu3']     = float(param_str[count])
    count += 1
    # Correctness parameter
    fixed_params['eps_c']   = float(param_str[count])
    count += 1
    # Secrecy parameter
    fixed_params['eps_s']   = float(param_str[count])
    count += 1
    # Probability of an after-pulse event
    fixed_params['Pap']     = float(param_str[count])
    count += 1
    # Transmission source repetition rate (Hz)
    fixed_params['Rrate']   = float(param_str[count])
    count += 1
    # Number of identical satellite overpasses
    fixed_params['NoPass']  = float(param_str[count])
    count += 1
    # Minimum elevation parameter (degs) minElev can be a float or None
    if str_is_None(param_str[count]):
        fixed_params['minElev'] = None
    else:
        fixed_params['minElev'] = float(param_str[count])
    count += 1
    # Angle to shift centre of transmission window from zenith (degs)
    fixed_params['shift0']  = float(param_str[count])
    count += 1
    # Transmission efficiency (set to unity)
    #fixed_params['eta']  = float(param_str[count])
    fixed_params['eta']  = 1.0
    #count += 1
    
    ###########################################################################
    # Iterable parameters
    ###########################################################################
    iter_params = {}
    # Maximum elevation of an orbit (degs)
    iter_params['theta_max'] = list_str_to_float(param_str[count])
    count += 1
    # Probabilty of an extraneous count
    iter_params['Pec']       = list_str_to_float(param_str[count])
    count += 1
    # Intrinsic Quantum Bit Error Rate
    iter_params['QBERI']     = list_str_to_float(param_str[count])
    count += 1
    # Time window half duration (s) - start, stop, step
    iter_params['dt']        = list_str_to_int(param_str[count])
    count += 1
    # Excess loss (dB) - start, stop, No steps
    iter_params['ls']        = list_str_to_float(param_str[count])
    count += 1
    
    ###########################################################################
    # Channel loss parameters
    ###########################################################################
    loss_params = {}
    # Read losses from file? Otherwise generate
    loss_params['tReadLoss'] = str2bool(param_str[count])
    count += 1
    if loss_params['tReadLoss']:
        # Path to loss data file
        loss_params['loss_path'] = strip_quote_input(param_str[count])
        count += 1
        # Name of loss data file
        loss_params['loss_file'] = strip_quote_input(param_str[count])
        count += 1
        # Column to read loss data from
        loss_params['loss_col']  = int(param_str[count])
    else:
        # Write losses to file?
        loss_params['tWriteLoss'] = str2bool(param_str[count])
        count += 1
        # Name of atmospheric data file (default is empty string)
        loss_params['atm_file'] = strip_quote_input(param_str[count])
        count += 1
        # Constant intrinsic system loss (dB)
        loss_int = float(param_str[count])
        # Convert to efficiency
        loss_params['eta_int']  = 10**(-loss_int/10.0)
        count += 1
        # Transmitter altitude (m)
        loss_params['h_T']       = float(param_str[count])
        count += 1
        # Receiver altitude (m)
        loss_params['h_R']       = float(param_str[count])
        count += 1
        # Transmitter aperture radius (m)
        loss_params['aT']        = float(param_str[count])
        count += 1
        # Receiver aperture radius (m)
        loss_params['aR']        = float(param_str[count])
        count += 1
        # Gaussian beam waist at focus (m) 
        loss_params['w0']        = float(param_str[count])
        count += 1
        # Wavelength of the transmitted light (nm)
        loss_params['wvl']       = float(param_str[count])
        count += 1
        # Radius of the Earth (m)
        loss_params['R_E']       = float(param_str[count])
        
    count += 1
        
    ###########################################################################
    # Output parameters
    ###########################################################################
    out_params, count = convert_str_params_out(param_str, count)
    
    ###########################################################################
    # Store parameter sets in a dictionary
    ###########################################################################
    params          = {}
    params['opt']   = opt_params
    params['fixed'] = fixed_params
    params['iter']  = iter_params
    params['loss']  = loss_params
    params['out']   = out_params
    return params

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def convert_str_params_adv(param_str):
    """
    Converts advanced calculation and optimiser parameters from string to values
    and returns as dictionary.

    Parameters
    ----------
    param_str : string, list
        List of string parameters read from file.

    Raises
    ------
    ValueError
        If the number of string parameters is less than the minimum expected.

    Returns
    -------
    params : mixed, dictionary
        Advanced parameter values.

    """
    # nMinParams = 12
    # nParams    = len(param_str) # Total number of lines in the input file
    # if nParams < nMinParams:
    #     raise ValueError('nParams = {} < nMinParams = {}'.format(nParams,nMinParams))
    count = 0
    ###########################################################################
    # Calculation parameters
    ###########################################################################
    calc_params = {}
    # The type of tail bounds to use for estimating statistical fluctuations
    # in the count statistics.
    calc_params['bounds']     = strip_quote_input(param_str[count])
    count += 1
    # The method for estimating the number of bits sacrificed for error
    # correction.
    calc_params['funcEC']     = strip_quote_input(param_str[count])
    count += 1
    # The efficiency factor when estimating the number of error correction bits.
    calc_params['fEC']        = float(param_str[count])
    count += 1
    # Numerical value to use when denominator values are potentially zero.
    calc_params['num_zero']   = float(param_str[count])
    count += 1

    ###########################################################################
    # Optimiser parameters
    ###########################################################################
    opt_params, count = convert_str_params_opt(param_str,count)
    
    ###########################################################################
    # Other parameters?
    ###########################################################################
    # other_params = {}
    # other_params['']     = param_str[count]
    # count += 1
    
    ###########################################################################
    # Store parameter sets in a dictionary
    ###########################################################################
    params          = {}
    params['calc']  = calc_params
    params['opt']   = opt_params
    # params['other'] = other_params
    return params

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def default_params_adv():
    """
    Returns default advanced calculation and optimiser parameter values.

    Returns
    -------
    params : mixed, dictionary
        Default advanced parameter values.

    """
    ###########################################################################
    # Calculation parameters
    ###########################################################################
    calc_params = {}
    # The type of tail bounds to use for estimating statistical fluctuations
    # in the count statistics.
    calc_params['bounds']     = boundOpts[0]
    # The method for estimating the number of bits sacrificed for error
    # correction.
    calc_params['funcEC']     = errcorrOpts[0]
    # The efficiency factor when estimating the number of error correction bits.
    calc_params['fEC']        = 1.16    
    # Numerical value to use when denominator values are potentially zero.
    calc_params['num_zero']   = 1.0e-10
    
    ###########################################################################
    # Optimiser parameters
    ###########################################################################
    opt_params = default_params_opt()
    
    ###########################################################################
    # Store parameter sets in a dictionary
    ###########################################################################
    params          = {}
    params['calc']  = calc_params
    params['opt']   = opt_params
    return params

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def check_params(main_params,adv_params):
    """
    Check that the main and advanced parameters are valid and update any values
    that are inconsistent.

    Parameters
    ----------
    main_params : mixed, dictionary
        Main calculation input parameters.
    adv_params : mixed, dictionary
        Advanced calculation input parameters.

    Raises
    ------
    TypeError
        For parameters with an invalid type.
    ValueError
        For parameters with an invalid value.

    Returns
    -------
    main_params : mixed, dictionary
        Updated main calculation input parameters.
    adv_params : mixed, dictionary
        Updated advanced calculation input parameters.

    """
    ###########################################################################
    # Check main parameters
    ###########################################################################
    # Optimisable parameters
    ###########################################################################
    main_params['opt']['tInit'] = False # Default value
    p_list = ['Px','P1','P2','mu1','mu2']
    # Loop over optimisable parameters
    for p in p_list:
        for ii in range(0,2,1):
            # Check logical/boolean parameters
            if type(main_params['opt'][p][ii]) is not bool:
                raise TypeError('{}[{}] = {} {}'.format(p,ii,main_params['opt'][p][ii],
                                                        type(main_params['opt'][p][ii])))
        if main_params['opt'][p][1]:
            # Set to True if any parameter has a specified initial value
            main_params['opt']['tInit'] = True
        for ii in range(2,5,1):
            # Check all values are in the range [0:1]
            if main_params['opt'][p][ii] is not None:
                if main_params['opt'][p][ii] < 0 or main_params['opt'][p][ii] > 1:
                    raise ValueError('{}[{}] = {}'.format(p,ii,main_params['opt'][p][ii]))
        # Check if optimised and count
        Nopt = 0
        if main_params['opt'][p][0]:
            Nopt += 1
            # Check upper bound is greater than lower bound
            if main_params['opt'][p][4] < main_params['opt'][p][3]:
                print('{} = {} < {}'.format(p,main_params['opt'][p][4],main_params['opt'][p][3]))
                raise ValueError('{}[{}] = {} < {}'.format(p,4,main_params['opt'][p][4],
                                                           main_params['opt'][p][3]))
            # Check if initialised
            if main_params['opt'][p][1]:
                # Check intial value is greater than lower bound
                if main_params['opt'][p][2] < main_params['opt'][p][3]:
                    print('{} = {} < {}'.format(p,main_params['opt'][p][2],main_params['opt'][p][3]))
                    raise ValueError('{}[{}] = {} < {}'.format(p,2,main_params['opt'][p][2],
                                                               main_params['opt'][p][3]))
                # Check intial value is less than upper bound
                if main_params['opt'][p][2] > main_params['opt'][p][4]:
                    print('{} = {} > {}'.format(p,main_params['opt'][p][2],main_params['opt'][p][4]))
                    raise ValueError('{}[{}] = {} > {}'.format(p,2,main_params['opt'][p][2],
                                                               main_params['opt'][p][4]))

    # Store the number of optimised parameters
    main_params['opt']['Nopt'] = Nopt

    ###########################################################################
    # Fixed parameters
    ###########################################################################
    p_list = ['mu3','eps_c','eps_s','Pap']
    # Loop over parameters
    for p in p_list:
        # Check parameters are in the range [0:1]
        if main_params['fixed'][p] < 0 or main_params['fixed'][p] > 1:
            raise ValueError('{} = {}'.format(p,main_params['fixed'][p]))
    
    p_list = ['Rrate','NoPass']
    # Loop over parameters
    for p in p_list:
        # Check parameters are positive values
        if main_params['fixed'][p] <= 0:
            raise ValueError('{} = {}'.format(p,main_params['fixed'][p]))

    p_list = ['minElev','shift0']
    # Loop over parameters
    for p in p_list:
        # Check parameters are in the range [0:90]
        if main_params['fixed'][p] < 0 or main_params['fixed'][p] > 90:
            raise ValueError('{} = {}'.format(p,main_params['fixed'][p]))
    
    ###########################################################################
    # Iterable parameters
    ###########################################################################
    # Check theta_max values
    for ii, tm in enumerate(main_params['iter']['theta_max']):
        # Check values are within the range [0:90]
        if tm < 0 or tm > 90:
            raise ValueError('theta_max[{}] = {}'.format(ii,tm))
    main_params['iter']['theta_max'] = np.radians(main_params['iter']['theta_max'])

    # Check probability of extraneous counts
    for ii, Pec in enumerate(main_params['iter']['Pec']):
        # Check values are within the range [0:1]
        if Pec < 0 or Pec > 1:
            raise ValueError('Pec[{}] = {}'.format(ii,Pec))
            
    # Check Intrinsic Quantum Bit Error Rate
    for ii, QBERI in enumerate(main_params['iter']['QBERI']):
        # Check valeus are positive
        if QBERI < 0:
            raise ValueError('QBERI[{}] = {}'.format(ii,QBERI))
    
    # Check dt range list has three values
    if len(main_params['iter']['dt']) != 3:
        raise ValueError('len(dt) = {}'.format(len(main_params['iter']['dt'])))
    # Check dt range start value is greater than zero
    if main_params['iter']['dt'][0] < 0:
        raise ValueError('dt[0] = {}'.format(len(main_params['iter']['dt'][0])))
    # Check dt range stop value is greater than zero
    if main_params['iter']['dt'][1] < 0:
        raise ValueError('dt[1] = {}'.format(len(main_params['iter']['dt'][1])))

    # Check ls range list has three values    
    if len(main_params['iter']['ls']) != 3:
        raise ValueError('len(ls) = {}'.format(len(main_params['iter']['ls'])))
    # Force No of steps to be an integer >= 1
    main_params['iter']['ls'][2] = int(max(1,main_params['iter']['ls'][2]))
    # Only take one ls value if the minimum and maximum values are the same
    if main_params['iter']['ls'][0] == main_params['iter']['ls'][1]:
        main_params['iter']['ls'][2] = 1
    # If start and stop are different use a minimum of 2 steps
    if main_params['iter']['ls'][0] != main_params['iter']['ls'][1]:
        main_params['iter']['ls'][2] = max(2,main_params['iter']['ls'][2])
    
    ###########################################################################
    # Channel loss parameters
    ###########################################################################
    # Read losses from file? Otherwise generate
    if type(main_params['loss']['tReadLoss']) is not bool:
        raise TypeError('tReadLoss = {} {}'.format(main_params['loss']['tReadLoss'],
                                                   type(main_params['loss']['tReadLoss'])))
    else:
        if not main_params['loss']['tReadLoss']:
            # Constant intrinsic system loss (db)
            if main_params['loss']['eta_int'] < 0:
                raise ValueError('eta_int = {}'.format(main_params['loss']['eta_int']))
            
            p_list = ['h_T','h_R']
            # Loop over parameters
            for p in p_list:
                # Check parameters are positive
                if main_params['loss'][p] < 0:
                    raise ValueError('{} = {}'.format(p,main_params['loss'][p]))

            p_list = ['aT','aR','w0','wvl']
            # Loop over parameters
            for p in p_list:
                # Check parameters are graeter than zero
                if main_params['loss'][p] <= 0:
                    raise ValueError('{} = {}'.format(p,main_params['loss'][p]))

    ###########################################################################
    # Check the advanced parameters
    ###########################################################################
    # Check the advanced calculation parameters
    ###########################################################################
    # Check bounds are valid
    if adv_params['calc']['bounds'].lower() not in (p.lower() for p in boundOpts):
        raise ValueError('bounds = {}'.format(adv_params['calc']['bounds']))
    # Check error correction function is valid
    if adv_params['calc']['funcEC'].lower() not in (p.lower() for p in errcorrOpts):
        raise ValueError('funcEC = {}'.format(adv_params['calc']['funcEC'])) 
    # The efficiency factor when estimating the number of error correction bits.
    if adv_params['calc']['fEC'] < 1.0:
        raise ValueError('fEC = {} < 1.0'.format(adv_params['calc']['fEC']))
    # # Numerical value to use when denominator values are potentially zero.
    if adv_params['calc']['num_zero'] <= 0 or adv_params['calc']['num_zero'] >= 1:
        raise ValueError('num_zero = {}'.format(adv_params['calc']['num_zero']))
        
    ###########################################################################
    # Check the advanced optimiser parameters
    ###########################################################################
    check_params_opt(adv_params['opt'])

    ###########################################################################
    # Check for inconsistent parameters and introduce any new values
    ###########################################################################
    if adv_params['calc']['bounds'].lower() == 'asymptotic':
        # Asymptotic bounds assume the following parameters
        main_params['fixed']['NoPass'] = 1
        adv_params['calc']['fEC']      = 'block'
    # Introduce new parameter that defines the total number of pulses
    main_params['fixed']['Npulse'] = main_params['fixed']['Rrate']*main_params['fixed']['NoPass']

    return main_params, adv_params