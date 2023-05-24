# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 11:12:17 2022

@author: Duncan McArthur
"""

from os.path import join
from time import (perf_counter, process_time, strftime)

import numpy as np

__all__ = ['get_data_header','getOptData','writeDataCSV','sort_data',
           'write_data']

###############################################################################

def get_data_header(protocol):
    """
    Get the output comma-separated data header. This header is used to size the
    output arrays.

    Parameters
    ----------
    protocol : str
        Name used for protocol within SatQuMA.

    Returns
    -------
    header : str
        Comma-separated string of data column labels for output arrays.

    """
    if protocol.lower() == 'aBB84-WCP'.lower():
        # Header for CSV file: Data columns
        # header = "SysLoss,dt,SKL,QBER,phiX,nX,nZ,mX,lambdaEC,sX0,sX1,vZ1,sZ1," + \
        #          "mean photon no.,QBERI,Pec,Pap,NoPass,Rrate,eps_c,eps_s," + \
        #          "PxA,PxB,P1,P2,P3,mu1,mu2,mu3,minElev (deg),maxElev (deg)," + \
        #          "shiftElev (deg),ls (dB)"
        header = "dt (s),ls (dB),QBERI,Pec,maxElev (deg),SKL (b),QBER,phiX," +\
                  "nX,nZ,mX,lambdaEC,sX0,sX1,vZ1,sZ1,mean photon no.,PxA," +\
                  "PxB,P1,P2,P3,mu1,mu2,mu3,eps_c,eps_s,Pap,NoPass," +\
                  "fs (Hz),minElev (deg),shiftElev (deg),SysLoss (dB)"
        return header
    else:
        print('There is no header for the requested protocol')
        return None
    
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
    try:
        # Print to the specified file
        print('Saving',message,'in file:',filename)
        np.savetxt(filename,data,delimiter=',',header=out_head)
    except PermissionError:
        # Add a timestamp to make a unique filename if there is a permission
        # error
        print(' > PermissionError: adding timestamp to filename')
        timestamp = strftime('.%Y-%m-%d_%H.%M.%S')
        filename = join(outpath, outfile + timestamp + '.csv')
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
            print('Error! Sort tag not recognised:',sort_tags[ii])
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

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def write_data(out_params,tOptimise,ni,ci,header,opt_head,outfile,fulldata,
               multidata,optdata):
    """
    Write data arrays to comma separated value (csv) format files.

    Parameters
    ----------
    out_params : dict
        Dictionary of parameters related to outputs.
    tOptimise : bool
        Flag to control parameter optimisation.
    ni : int, array-like
        Number of calculations per iterable parameter.
    ci : int, array-like
        Loop counter for each iterable parameter.
    header : str
        Data column headers for main output array.
    opt_head : str
        Data column headers for optimiser metric array.
    outfile : str
        Name for output file.
    fulldata : float, array
        Array to store main calculation outputs.
    optdata : float, array
        Array to store optimiser metric data.
    multidata : float, array
        Array to store multi-output calculation data.

    Returns
    -------
    multidata : float, array
        Array to store multi-output calculation data.

    """
    if out_params['tPrint']:
        print('-'*60,'\n')
    if out_params['tFullData']:
        # Write out data in CSV format
        writeDataCSV(fulldata[:ni[3]*ni[4],:],out_params['out_path'],outfile,
                     header,'full loss & time data')
    if (tOptimise and out_params['tMetrics']):
        # Write optimiser metrics
        #print(res.keys()) # Prints available outputs for the object res
        writeDataCSV(optdata,out_params['out_path'],outfile+'_metrics',opt_head,
                     'optimisation metrics')
    if (out_params['tdtOptData'] and ni[4] > 1):
        # Sort data by SKL per SysLoss
        sortdata = sort_data(fulldata[:ni[3]*ni[4],:],header,
                             ['SKL (b)','SysLoss (dB)'],[True,False])
        # Store optimal dt data per xi, Pec, and QBERI
        cm0 = ci[0]*(ni[1]*ni[2]*ni[3]) + ci[1]*(ni[2]*ni[3]) + ci[2]*(ni[3])
        cm1 = cm0 + ni[3]
        multidata[cm0:cm1,:] = sortdata[::ni[4],:]
    if out_params['tPrint']:
        print('-'*60,'\n')
    return multidata

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def writeMultiData(out_dict,header,multidata):
    """
    Write a multi-output data file containing dt optimised data for each of the
    other iterable parameters.

    Parameters
    ----------
    out_dict : dict
        Dictionary of parameters related to outputs.
    header : str
        Data column headers for main output array.
    multidata : float, array
        Array that stores multi-output calculation data.

    Returns
    -------
    None.

    """
    if out_dict['tdtOptData']:
        # Filename for multi-optimal data
        multifile = out_dict['out_base'] + '_multi-theta-Pec-QBERI-ls'
        # Write sorted data for all xi, Pec, and QBERIs
        writeDataCSV(multidata,out_dict['out_path'],multifile,header,
                     'all systems optimal data')
    return None
        
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def out_arrays(ni,header,opt_head,tOptimise,tMultiOpt):
    """
    Initialise arrays based on user requested outputs.

    Parameters
    ----------
    ni : int, array-like
        Number of calculations per iterator.
    header : str
        Data column headers for main output array.
    opt_head : str
        Data column headers for optimiser metric array.
    tOptimise : bool
        Flag to control parameter optimisation.
    tMultiOpt : bool
        Flag to control use of multi-output data file.

    Returns
    -------
    fulldata : float, array
        Array to store main calculation outputs.
    optdata : float, array
        Array to store optimiser metric data.
    multidata : float, array
        Array to store multi-output calculation data.

    """
    # Initialise a data storage array: shape(No. of data runs, No. of metrics)   
    fulldata = np.empty((ni[3]*ni[4],len(header.split(","))))
    
    #if tMultiOpt and (ni[0]*ni[1]*ni[2]*ni[3]) > 1:
    if tMultiOpt:
        multidata = np.empty((ni[0]*ni[1]*ni[2]*ni[3],len(header.split(","))))
    else:
        multidata = None
    
    if tOptimise:
        # Initialise a data storage array for optimiser metrics
        optdata = np.empty((ni[3]*ni[4],len(opt_head.split(","))))
    else:
        optdata = None
    return fulldata, optdata, multidata

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def printLog(string,tLog=False,file='out.log'):
    """
    Print a string to Stdout and also append to a log file if requested.

    Parameters
    ----------
    string : str
        String to write.
    tLog : bool, optional
        Write to log file? The default is False.
    file : str, optional
        Path and name of log file. The default is 'out.log'.

    Returns
    -------
    None.

    """
    if tLog:

        try:
            with open(file, "a") as f:
                f.write(string+'\n')
        except FileNotFoundError:
            msg = "Sorry, the file "+ file + "does not exist."
            print(msg) # Sorry, the file John.txt does not exist.
    print(string)
    return None
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
def get_timings():
    """
    Get time floats from both the system clock and CPU timer.

    Returns
    -------
    tc : float
        Clock timer (s).
    tp : TYPE
        CPU timer (s).

    """
    tc = perf_counter() # Start clock timer
    #tc = perf_counter_ns()/1e9 # Start clock timer
    tp = process_time() # Start CPU timer
    return tc, tp

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def format_seconds_to_ddhhmmss(seconds):
    """
    Convert a time in seconds on into day:hour:minute:second format string

    Parameters
    ----------
    seconds : float
        Time duration (s).

    Returns
    -------
    str
        Formatted time string.

    """
    days = seconds // (60*60*24)
    seconds %= seconds // (60*60*24)
    hours = seconds // (60*60)
    seconds %= (60*60)
    minutes = seconds // 60
    seconds %= 60
    return "{:02.0f}:{:02.0f}:{:02.0f}:{:02.0f} (dd:hh:mm:ss)".format(days,
                                                                      hours,
                                                                      minutes,
                                                                      seconds)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def format_seconds_to_hhmmss(seconds):
    """
    Convert a time in seconds on into hour:minute:second format string

    Parameters
    ----------
    seconds : float
        Time duration (s).

    Returns
    -------
    str
        Formatted time string.

    """
    hours = seconds // (60*60)
    seconds %= (60*60)
    minutes = seconds // 60
    seconds %= 60
    return "{:02.0f}:{:02.0f}:{:02.0f} (hh:mm:ss)".format(hours,minutes,seconds)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def format_subseconds(seconds):
    """
    Convert a time in seconds into the closest subsecond units. 

    Parameters
    ----------
    seconds : float
        Time duration (s).

    Returns
    -------
    str
        Formatted time string.

    """
    from math import (log10, floor)
    expabs = abs(floor(log10(seconds))) # Get magnitude of floored exponent
    if expabs <= 3:
        # milliseconds
        return "{:3.2f} (ms)".format(seconds*10**3)
    elif expabs <= 6:
        # microseconds
        return "{:3.2f} (µs)".format(seconds*10**6)
    elif expabs <= 9:
        # nanoseconds
        return "{:3.2f} (ns)".format(seconds*10**9)
    elif expabs <= 12:
        # picoseconds
        return "{:3.2f} (ps)".format(seconds*10**12)
    elif expabs <= 15:
        # femtoseconds
        return "{:3.2f} (fs)".format(seconds*10**15)
    else:
        # femtoseconds
        return "{} (fs)".format(seconds*10**15)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def format_time(seconds):
    """
    Convert a time in seconds into the appropriate formatted string.
      - susbseconds
      - hh:mm:ss
      - dd:hh:mm:ss

    Parameters
    ----------
    seconds : float
        Time duration (s).

    Returns
    -------
    str
        Formatted time string.

    """
    if seconds == 0:
        return '-.- (s)'
    elif seconds < 1:
        # Subseconds
        return format_subseconds(seconds)
    else:
        if seconds >= 86400:
            # day:hour:min:sec
            return format_seconds_to_ddhhmmss(seconds)
        else:
            # hour:min:sec
            return format_seconds_to_hhmmss(seconds)