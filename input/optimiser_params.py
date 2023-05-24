# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 15:50:34 2021

@author: Duncan McArthur
"""

from .parse_input_string import (str2bool, strip_quote_input)

__all__ = ['convert_str_params_opt','default_params_opt','check_params_opt']

###############################################################################

# List of available optimiser methods/algorithms from scipy.optimize
opt_methods = ['COBYLA','SLSQP','trust-constr']

###############################################################################

def convert_str_params_opt(param_str,count):
    """
    Converts input strings into advanced optimiser parameter values and returns
    as a dictionary.

    Parameters
    ----------
    param_str : string, list
        List of string parameters.
    count : integer
        Index counter.

    Raises
    ------
    ValueError
        If specified optimiser algorithm is not recognised.

    Returns
    -------
    opt_params : mixed, dictionary
        Dictionary of optimiser parameter values.
    count : integer
        Updated index counter.

    """
    opt_params = {} # Initialise dictionary
    # Minimum No. of optimisations to strive for
    opt_params['NoptMin']     = int(param_str[count])
    count += 1
    # Maximum No. of optimisations (not used)
    opt_params['NoptMax']     = int(param_str[count])
    count += 1
    # Stop optimizing if the first NoptMin return SKL = 0?
    opt_params['tStopZero']   = str2bool(param_str[count])
    count += 1
    # Stop after NoptMin optimizations if SKL improved?
    opt_params['tStopBetter'] = str2bool(param_str[count])
    count += 1
    # Optimiser alogortihm (see scipy.optimize)
    opt_params['method']      = strip_quote_input(param_str[count])
    count += 1
    # Max No. of iterations
    opt_params['Nmax']        = int(param_str[count])
    count += 1
    # Get method specific parameters
    if opt_params['method'].upper() == 'COBYLA':
        opt_params['method'] = opt_params['method'].upper()
        # Constraint absolute tolerance.
        opt_params['ctol']    = float(param_str[count])
        count += 1
        # Reasonable initial changes to the variables.
        opt_params['rhobeg'] = float(param_str[count])
        count += 1
    elif opt_params['method'].upper() == 'SLSQP':
        opt_params['method'] = opt_params['method'].upper()
        # Precision goal for the value of f in the stopping criterion.
        opt_params['ftol'] = float(param_str[count])
        count += 1
        # Step size used for numerical approximation of the Jacobian.
        opt_params['eps']  = float(param_str[count])
        count += 1
    elif opt_params['method'].lower() == 'trust-constr':
        opt_params['method'] = opt_params['method'].lower()
        # Tolerance to terminate by change of independent variable(s)
        opt_params['xtol']      = float(param_str[count])
        count += 1
        # Tolerance to terminate Lagrangian gradient
        opt_params['gtol']      = float(param_str[count])
        count += 1
        # Threshold on the barrier parameter for termination
        opt_params['btol']      = float(param_str[count])
        count += 1
        # Initial constraint penalty parameter (default = 1.0)
        opt_params['const_pen'] = float(param_str[count])
        count += 1
        # Initial trust radius (default = 1.0)
        opt_params['tr_rad']    = float(param_str[count])
        count += 1
        # Initial barrier parameter (default = 0.1)
        opt_params['barr_par']  = float(param_str[count])
        count += 1
        # Initial barrier tolerance for barrier sub- (default = 0.1)
        opt_params['barr_tol']  = float(param_str[count])
        count += 1
    else:
        print('Optimiser method not recognised:',opt_params['method'])
        print('Select one of:',*["'{0}'".format(m) for m in opt_methods])
        raise ValueError('opt_method = {}'.format(opt_params['method']))
    return opt_params, count

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def default_params_opt():
    """
    Returns default parameter values for the scipy.optimize algorithm and
    general optimisation process. 

    Raises
    ------
    ValueError
        If specified optimiser algorithm is not recognised.

    Returns
    -------
    opt_params : mixed, dictionary
        Dictionary of optimiser parameter values.

    """
    opt_params = {} # Initialise dictionary
    # Optimiser alogortihm (see scipy.optimize)
    opt_params['method']      = opt_methods[0]
    # Minimum No. of optimisations to strive for
    opt_params['NoptMin']     = 10
    # Maximum No. of optimisations (not used)
    opt_params['NoptMax']     = 1000
    # Stop optimizing if the first NoptMin return SKL = 0?
    opt_params['tStopZero']   = True
    # Stop after NoptMin optimizations if SKL improved?
    opt_params['tStopBetter'] = True
    # Max No. of iterations
    opt_params['Nmax']        = 1000
    # Get method specific parameters
    if opt_params['method'].upper() == 'COBYLA':
        # Constraint absolute tolerance.
        opt_params['ctol']    = 1.0e-12
        # Reasonable initial changes to the variables.
        opt_params['rhobeg'] = 0.002
    elif opt_params['method'].upper() == 'SLSQP':
        # Precision goal for the value of f in the stopping criterion.
        opt_params['ftol'] = 1.0e-12
        # Step size used for numerical approximation of the Jacobian.
        opt_params['eps']  = 1.0e-7
    elif opt_params['method'].lower() == 'trust-constr':
        # Tolerance to terminate by change of independent variable(s)
        opt_params['xtol']      = 1.0e-8
        # Tolerance to terminate Lagrangian gradient
        opt_params['gtol']      = 1.0e-10
        # Threshold on the barrier parameter for termination
        opt_params['btol']      = 1.0e-8
        # Initial constraint penalty parameter (default = 1.0)
        opt_params['const_pen'] = 1.0
        # Initial trust radius (default = 1.0)
        opt_params['tr_rad']    = 1.0
        # Initial barrier parameter (default = 0.1)
        opt_params['barr_par']  = 0.1
        # Initial barrier tolerance for barrier sub- (default = 0.1)
        opt_params['barr_tol']  = 0.1
    else:
        print('Optimiser method not recognised:',opt_params['method'])
        print('Select one of:',*["'{0}'".format(m) for m in opt_methods])
        raise ValueError('opt_method = {}'.format(opt_params['method']))
    return opt_params

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def check_params_opt(opt_params):
    """
    Check that the optimiser values are valid.

    Parameters
    ----------
    opt_params : mixed, dictionary
        Optimiser parameters.

    Returns
    -------
    None.

    """
    
    # List of values to check
    p_list = ['NoptMin','NoptMax','Nmax']
    for p in p_list:
        # Must be greater than zero
        if opt_params[p] < 1:
            raise ValueError('{} = {}'.format(p,opt_params[p]))

    # List of logical values to check
    p_list = ['tStopZero','tStopBetter']
    for p in p_list:
        # Must not be None
        if type(opt_params[p]) is not bool:
            raise TypeError('{} = {} {}'.format(p,opt_params[p],type(opt_params[p])))

    # Get method specific parameters
    if opt_params['method'].upper() == 'COBYLA':
        # List of values to check
        p_list = ['ctol','rhobeg']
        for p in p_list:
            # Must be greater than zero
            if opt_params[p] <= 0:
                raise ValueError('{} = {}'.format(p,opt_params[p]))
    elif opt_params['method'].upper() == 'SLSQP':
        # List of values to check
        p_list = ['ftol','eps']
        for p in p_list:
            # Must be greater than zero
            if opt_params[p] <= 0:
                raise ValueError('{} = {}'.format(p,opt_params[p]))
    elif opt_params['method'].lower() == 'trust-constr':
        # List of values to check
        p_list = ['xtol','gtol','btol','const_pen','tr_rad','barr_par','barr_tol']
        for p in p_list:
            # Must be greater than zero
            if opt_params[p] <= 0:
                raise ValueError('{} = {}'.format(p,opt_params[p]))
    else:
        print('Optimiser method not recognised:',opt_params['method'])
        print('Select one of:',*["'{0}'".format(m) for m in opt_methods])
        raise ValueError('opt_method = {}'.format(opt_params['method']))