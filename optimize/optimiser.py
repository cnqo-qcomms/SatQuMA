# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 17:02:45 2021

@author: Duncan McArthur
"""

#from scipy.optimize import minimize
import numpy as np
from sys import float_info

from key.protocols.init_efficient_BB84 import get_x_bounds
from output.outputs import get_data_header

##############################################################################

# Extract the smallest float that the current system can:
num_min = float_info.epsilon # round (relative error due to rounding)

opt_methods = ['COBYLA','SQLSP','trust-constr']

##############################################################################

def opt_arrays(iter_dict):
    """
    Initialises the arrays used to size and count loops as well as store
    intial parameter data

    Parameters
    ----------
    iter_dict : dict
        Dictionary of parameters for the iterable parameters.

    Returns
    -------
    ni : int, array
        Number of calculations per iterator.
    ci : int, array
        Loop counter array.
    x0i : float, array
        Initial parameters for each loop.

    """
    ls_range   = iter_dict['ls']
    dt_range   = iter_dict['dt']
    theta_list = iter_dict['theta_max']
    Pec_list   = iter_dict['Pec']
    QBERI_list = iter_dict['QBERI']
    
    #*************************************************************************
    # Initialise loop ranges
    #*************************************************************************
    # Set the number of outputs to store
    ni = np.empty((5,),dtype=np.int16)
    
    ni[0] = len(theta_list) # Number of xi values to calculate for
    ni[1] = len(Pec_list)   # Number of Pec values to calculate for
    ni[2] = len(QBERI_list) # Number of QBERI values to calculate for
    
    # Determine number of calculations requested
    ni[3] = ls_range[2] # No. of loss values
    if (dt_range[0] == dt_range[1] or dt_range[2] == 0):
        ni[4] = 1
    else:
        ni[4] = int((dt_range[1] - dt_range[0]) / float(dt_range[2])) + 1

    x0i = np.empty((5,*ni)) # Array to store initialisation parameters
    ci  = np.empty((5,),dtype=np.int16) # Array to count the various loops
    return ni, ci, x0i

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def set_bounds(opt_dict,protocol):
    """
    Set upper and lower boundary values for numerical optimisation of protocol
    parameters.

    Parameters
    ----------
    opt_dict : dict
        Dictionary of parameters related to optimisation.
    protocol : str
        Name of protocol.

    Returns
    -------
    bounds : obj
        Scipy bounds object containing upper/lower bounds.

    """
    from sys import float_info
    # Extract the smallest float that the current system can:
    num_min = float_info.epsilon # round (relative error due to rounding)
    
    # Initialise lower/upper bound arrays
    lb = np.zeros(opt_dict['Nopt'])
    ub = np.zeros(opt_dict['Nopt'])
    # set list of optimisable parameters
    
    ### Change to call to aBB84
    if opt_dict['Px'] == True:
        opt_list = ['PxA','P1','P2','mu1','mu2']
    else:
        opt_list = ['PxA','PxB','P1','P2','mu1','mu2']
    # Store bounds value sin arrays
    for ii, popt in enumerate(opt_list):
        # Is parameter optimised?
        if opt_dict[popt][0]:
            lb[ii] = opt_dict[popt][3] + num_min # Lower bound
            ub[ii] = opt_dict[popt][4] - num_min # Upper bound

    from scipy.optimize import Bounds
    # Store the parameters bounds as an object, as required by minimize() for
    # method='trust-constr'
    bounds = Bounds(lb, ub)          # Store upper and lower bounds
    return bounds
 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
    
def set_constraints(opt_dict,fixed_dict,adv_opt,protocol):
    """
    Set the optimisation constraints based on the protocol and user requests.

    Parameters
    ----------
    opt_dict : dict
        Dictionary of parameters related to optimisation.
    fixed_dict : dict
        Dictionary of fixed parameters related to SKL calculation.
    adv_opt : dict
        Dictionary of advanced calculation parameters.
    protocol : str
        Name of protocol.

    Raises
    ------
    ValueError
        Protocol name not recognised.

    Yields
    ------
    bounds : obj
        Scipy bounds object for upper & lower bounds.
    cons : obj
        Scipy object for optimisation constraints.
    options : dict
        Dictionary of optimiser parameters.
    x : float, array
        Optimised parameters initial values.
    xb : float, array
        Optimised parameters upper & lower bounds.

    """
    
    # if opt_dict['Nopt'] > 1:
    #     lb_l = np.zeros((opt_dict['Nopt']))
    #     ub_l = np.zeros((opt_dict['Nopt']))
    #     A_l  = np.zeros((opt_dict['Nopt'],opt_dict['Nopt']))
    #     if protocol.lower() == 'efficient-BB84'.lower():
    #         from .protocol_inputs.input_efficient_BB84 import protocol_constraints
    #         lu_bound, con_fac  = protocol_constraints()
    #         ### Change to call to aBB84
    #         if opt_dict['Px'] == True:
    #             opt_list = ['PxA','P1','P2','mu1','mu2']
    #         else:
    #             opt_list = ['PxA','PxB','P1','P2','mu1','mu2']
    #     else:
    #         raise ValueError('Protocol not recognised: {}'.format(protocol))
        
    #     count = 0
    #     for ii, popt in enumerate(opt_list):
    #         # Is parameter optimised?
    #         if opt_dict[popt][0]:
    #             lb_l[count] = lu_bound[0,ii]
    #             ub_l[count] = lu_bound[1,ii] # Upper bound
    #             for jj, popt2 in enumerate(opt_list):
    #                 if opt_dict[popt][0]:
    #                     A_l[ii,jj] = con_fac[ii,jj]
                        
    ##########################################################################
    
    method = adv_opt['method'] # Optimisation method
    mu3    = fixed_dict['mu3'] # Intensity of pulse 3 (fixed)
    
    x, xb = get_x_bounds(opt_dict,mu3,num_min)
    
    if not opt_dict['tOptimise']:
        return None, None, None, x, xb
    
    # Build bounds arrays for x = [Px,pk1,pk2,mu1,mu2]
    # Lower bounds for parameters
    lb = np.array([xb[0,0],xb[1,0],xb[2,0],xb[3,0],xb[4,0]]) + num_min
    # Upper bounds for parameters
    ub = np.array([xb[0,1],xb[1,1],xb[2,1],xb[3,1],xb[4,1]]) - num_min
    
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
        options = {'xtol': adv_opt['xtol'],
                   'gtol':  adv_opt['gtol'], 
                   'barrier_tol':  adv_opt['btol'],
                   'sparse_jacobian': None, 
                   'maxiter':  adv_opt['Nmax'],
                   'verbose': 0, 
                   'finite_diff_rel_step': None, 
                   'initial_constr_penalty':  adv_opt['const_pen'],
                   'initial_tr_radius':  adv_opt['tr_rad'], 
                   'initial_barrier_parameter':  adv_opt['barr_par'], 
                   'initial_barrier_tolerance':  adv_opt['barr_tol'], 
                   'factorization_method': None,
                   'disp': False}
    else:
        # Build inequality constraint dictionary for 'COBYLA' or 'SLSQP'
        cons_type = 'ineq' # C_j[x] >= 0
        def cons_fun(x):
            """
            Function that returns the constraints for a set of optimised 
            parameters x = [PxA,pk1,pk2,mu1,mu2]. Note, mu3 is passed to the
            optimiser as a fixed parameter when called.
            
            Applies the following 3 linear constraint inequality constraints:
                (1)         1 - pk1 - pk2 >= 0,
                (2) mu1 - mu2 - mu3 - eps >= 0,
                (3)       mu2 - mu3 - eps >= 0,
                where eps is an arbitrarily small number.

            Parameters
            ----------
            x : float, array-like (5,)
                Optimised parameters.

            Returns
            -------
            float, array-like (3,)
                Linear constraints.

            """
            return np.array([1 - x[1] - x[2],
                             x[3] - x[4] - mu3 - num_min,
                             x[4] - mu3 - num_min])
        
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        def cons_jac(x):
            """
            Returns the Jacobian of the constraints function.
            
            We have 3 constraints C[:] and 5 parameters in the array x[:].
            The retuned array has a row for each constraint and a column for
            each parameter.
            
            The returned value is given by y[ii,jj] = d C[ii] / d x[jj]

            Parameters
            ----------
            x : float, array-like (5,)
                Optimised parameters.

            Returns
            -------
            float, array-like (3,5)
                Jacobian of the constraints function.

            """
            # Jacobian of the linear constraint inequality function
            return np.array([[0,-1,-1,0,0],
                             [0,0,0,1,-1],
                             [0,0,0,0,1]])
        
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        # Define linear constraint dictionary
        lin_cons = {'type' : cons_type,
                    'fun'  : cons_fun,
                    'jac'  : cons_jac}
        
        if method == 'COBYLA':
            bounds = None # COBYLA doesn't support bounds
            
            # Build inequality constraint dictionaries for bounds
            cons_type = 'ineq' # C_j[x] >= 0
            def upper_fun(x):
                """
                Returns upper bounds as a constraint function for the 
                optimised parameters x[:].

                Parameters
                ----------
                x : float, array-like (5,)
                    Optimised parameters.

                Returns
                -------
                float, array-like (5,)
                    Upper bound as constraints.

                """
                # Upper bound inequality function
                return np.array([ub[0] - x[0],
                                 ub[1] - x[1],
                                 ub[2] - x[2],
                                 ub[3] - x[3],
                                 ub[4] - x[4]])
            
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
            def upper_jac(x):
                """
                Jacobian of the upper bound constraint array.
                
                We have 5 constraints C[:] and 5 parameters in the array x[:].
                The retuned array has a row for each constraint and a column 
                for each parameter.
            
                The returned value is given by y[ii,jj] = d C[ii] / d x[jj]

                Parameters
                ----------
                x : float, array-like (5,)
                Optimised parameters.

                Returns
                -------
                float, array-like (5,5)
                    Jacobian of the upper bound constraint function.

                """
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
            
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
            def lower_fun(x):
                """
                Returns lower bounds as a constraint function for the 
                optimised parameters x[:].

                Parameters
                ----------
                x : float, array-like (5,)
                    Optimised parameters.

                Returns
                -------
                float, array-like (5,)
                    Lower bound as constraints.

                """
                # Lower bound inequlity function
                return np.array([x[0] - lb[0],
                                 x[1] - lb[1],
                                 x[2] - lb[2],
                                 x[3] - lb[3],
                                 x[4] - lb[4]])
            
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
            def lower_jac(x):
                """
                Jacobian of the lower bound constraint array.
                
                We have 5 constraints C[:] and 5 parameters in the array x[:].
                The retuned array has a row for each constraint and a column 
                for each parameter.
            
                The returned value is given by y[ii,jj] = d C[ii] / d x[jj]

                Parameters
                ----------
                x : float, array-like (5,)
                Optimised parameters.

                Returns
                -------
                float, array-like (5,5)
                    Jacobian of the lower bound constraint function.

                """
                # Lower bound inequality Jacobian
                return np.array([[1,0,0,0,0],
                                 [0,1,0,0,0],
                                 [0,0,1,0,0],
                                 [0,0,0,1,0],
                                 [0,0,0,0,1]])
            
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
            # Define lower bound inequality dictionary
            lower = {'type' : cons_type,
                     'fun'  : lower_fun,
                     'jac'  : lower_jac}
            
            # Tuple of all constraint(bounds) dictionaries
            cons = (upper, lower, lin_cons)
            
            # Set specific optimiser options
            options = {'rhobeg': adv_opt['rhobeg'],
                       'maxiter': adv_opt['Nmax'], 
                       'disp': False, 
                       'catol': adv_opt['ctol']}
            
        elif method == 'SLSQP':
            # Singleton tuple of bounds
            cons = (lin_cons,)
            
            # Set specific optimiser options
            options = {'maxiter': adv_opt['Nmax'],
                       'ftol': adv_opt['ftol'],
                       'iprint': 1,
                       'disp': False,
                       'eps': adv_opt['eps'],
                       'finite_diff_rel_step': None}
        else:
            print('Optimiser method not recognised:',method)
            print('Select one of:',*["'{0}'".format(m) for m in opt_methods])
            exit(1)
    return bounds, cons, options, x, xb

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def out_heads(protocol,tOptimise,method):
    """
    Returns headers for main calculation data array and the optimiser metric 
    data array.

    Parameters
    ----------
    protocol : str
        Name for protocol in SatQuMA.
    tOptimise : bool
        Flag to control optimisation of parameters.
    method : str
        Optimisation method.

    Returns
    -------
    header : str
        Data column headers for main output array.
    opt_head : str
        Data column headers for optimiser metric array.

    """
    # Header for CSV file: Data columns
    header = get_data_header(protocol)
    
    if tOptimise:
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
            opt_head = ""
    else:
        opt_head = ""
    return header, opt_head