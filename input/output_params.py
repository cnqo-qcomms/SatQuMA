# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:41:10 2021

@author: Duncan McArthur
"""

from .parse_input_string import (str2bool, strip_quote_input)

__all__ = ['convert_str_params_out']

###############################################################################

def convert_str_params_out(param_str,count):
    """
    Converts input strings into output parameter values and returns as a 
    dictionary.

    Parameters
    ----------
    param_str : string, list
        List of string parameters.
    count : integer
        Index counter.

    Returns
    -------
    out_params : mixed, dictionary
        Dictionary of output parameter values.
    count : integer
        Updated index counter.

    """

    out_params = {} # Initialise dictionary
    # Print calculation data to StdOut?
    out_params['tPrint']     = str2bool(param_str[count])
    count += 1
    # Write full data arrays to file (.csv)?
    out_params['tFullData']  = str2bool(param_str[count])
    count += 1
    # Write dt optimised data arrays to file (.csv)?
    out_params['tdtOptData']  = str2bool(param_str[count])
    count += 1
    # Write optimiser metrics to file?
    out_params['tMetrics']   = str2bool(param_str[count])
    count += 1
    # Path to write output data files
    out_params['out_path']   = strip_quote_input(param_str[count])
    count += 1
    # Basename for all output data files
    out_params['out_base']   = strip_quote_input(param_str[count])
    count += 1
    # Add extra value to flag whether data should be written to file
    if any([out_params['tFullData'],out_params['tdtOptData'],
            out_params['tMetrics']]):
        out_params['tWriteFiles'] = True
    else:
        out_params['tWriteFiles'] = False
    return out_params, count