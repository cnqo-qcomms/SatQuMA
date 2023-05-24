# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 09:48:12 2021

@author: Duncan McArthur
"""

#import numpy as np
from sys import exit
from os.path import isfile
from .parse_input_string import (str2bool, uncomment_input, 
                                 split_input, input_from_list)

__all__ = ['read_from_file','get_param_str_list','read_input','read_input_adv',
           'get_input_adv','get_input']

##############################################################################

min_lines = 1
# List of available security protocols
list_protocols = ['aBB84-WCP']

##############################################################################

def read_from_file(filename):
    """
    Read in lines from a data file as string.

    Parameters
    ----------
    filename : string
        Input file.

    Raises
    ------
    ValueError
        If there is not at least min_lines in the file.

    Returns
    -------
    data : string, list
        List of input parameters as strings.
    nLines : integer
        Number of lines in data.

    """
    with open(filename, "r") as file:
        data   = file.readlines() # Read all data from the input file
    nLines = len(data)        # Total number of lines in the input file
    if (nLines < min_lines):
        print('Error! Input file {} does not have enough lines'.format(filename))
        raise ValueError('nLines = {}'.format(nLines))
    return data, nLines

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_param_str_list(data,nextLine,count):
    """
    Returns uncommented, delimiter (comma) separated data as a nested list of 
    strings.

    Parameters
    ----------
    data : string, list
        Lines from input file as string.
    nextLine : integer
        Data line tostart parsing from.
    count : integer
        Counts parameters.

    Returns
    -------
    param_list : string, list
        Parameters from input as strings.
    count : integer
        Updated parameter count.

    """
    param_list = []
    for thisLine in range(nextLine,len(data),1):
        # read data until end of file
        x = uncomment_input(data[thisLine],'#') # Remove comments - anything following a '#'
        if x:
            # Data string is not empty
            count += 1
            x = split_input(x,',') # Split input line by commas
            #print(str(count)+':',x)
            # Append parameter string(s) to list
            if len(x) > 1:
                # x is a list. Append as nested list
                param_list.append(x)
            else:
                # x is a single value. Append to list
                param_list.append(*x)
    return param_list, count

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def read_input(filename):
    """
    Read in lines from the main input file, parse, and return parameter values.

    Parameters
    ----------
    filename : string
        Name of input file.

    Raises
    ------
    ValueError
        If protocol name is not recognised.

    Returns
    -------
    x1 : string
        Protocol name.
    param_dict : mixed, dictionary
        Calculation parameters.

    """
    data, nLines = read_from_file(filename) # Read in data
    count        = 0 # Count parameter sets
    thisLine     = 0 # Count input file lines
       
    #######################################################################
    ### Parameter 1: Security protocol
    x1, thisLine, count = input_from_list(data,thisLine,nLines,count,
                                          list_protocols,'Security protocol')
    
    #######################################################################
    ### Read the rest of the parameters into a list
    param_list, count = get_param_str_list(data,thisLine + 1,count)
    
    #######################################################################
    ### Convert input strings into parameter values based on protocol
    if x1.lower() == 'aBB84-WCP'.lower():
        from .protocol_inputs.input_efficient_BB84 import convert_str_params
        param_dict = convert_str_params(param_list) # Dictionary of parameters
    else:
        raise ValueError('Protocol not recognised: {}'.format(x1))  
    return x1, param_dict

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def read_input_adv(filename,protocol):
    """
    Read in lines from an advanced input file, parse, and return parameter 
    values.

    Parameters
    ----------
    filename : string
        Name of input file.
    protocol : string
        Protocol name.

    Raises
    ------
    ValueError
        If protocol name is not recognised.

    Returns
    -------
    param_dict : mixed, dictionary
        Advanced calculation parameters.

    """
    data, nLines = read_from_file(filename) # Read in data
    count        = 0 # Count parameter sets
    thisLine     = 0 # Count input file lines
    
    #######################################################################
    ### Read parameters into a list
    param_list, count = get_param_str_list(data,thisLine,count)
    
    #######################################################################
    ### Convert parameter values based on protocol
    if protocol.lower() == 'aBB84-WCP'.lower():
        from .protocol_inputs.input_efficient_BB84 import convert_str_params_adv
        param_dict = convert_str_params_adv(param_list) # Dictionary of parameters
    else:
        raise ValueError('Protocol not recognised: {}'.format(protocol)) 
    return param_dict

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
def get_input_adv(protocol):
    """
    Returns default advanced parameters for a given protocol

    Parameters
    ----------
    protocol : string
        Protocol name.

    Raises
    ------
    ValueError
        If protocol name is not recognised.

    Returns
    -------
    param_dict : mixed, dictionary
        Default advanced parameters.

    """
    # Check for protocol
    if protocol.lower() == 'aBB84-WCP'.lower():
        from .protocol_inputs.input_efficient_BB84 import default_params_adv
        param_dict = default_params_adv() # Dictionary of parameters
    else:
        raise ValueError('Protocol not recognised: {}'.format(protocol)) 
    return param_dict

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
         
def get_input(filename,filename_adv='input-adv.txt'):
    """
    Primary function to retrieve user specified input parameters. Returns both
    main calculation and advanced parameters. Parameters are converted from
    string and verified before being returned as dictionaries.

    Parameters
    ----------
    filename : string
        Name of main parameter input file.
    filename_adv : string, optional
        Name of advanced parameter input file. The default is 'input-adv.txt'.

    Raises
    ------
    ValueError
        If protocol name is not recognised.

    Returns
    -------
    protocol : string
        Protocol name.
    main_params : mixed, dictionary
        Main calculation parameters.
    adv_params : mixed, dictionary
        Advanced calculation parameters.

    """
    ###########################################################################
    # Get main calculation parameters            
    if (isfile(filename)):
        print(' > Reading parameters from input file')
        protocol, main_params = read_input(filename)
    else:
        print('Input file not found')
        while True:
            sGenInput = input('Generate default inputfile? Else enter input manually. (y/n) ')
            tGenInput = str2bool(sGenInput)
            if (type(tGenInput) is bool):
                break
            else:
                print('User input not recognised:',sGenInput+'.','Please enter y or n.')
        if tGenInput:
            print(' > Generating default inputfile')
            print('TBD...')
            exit(1)
        else:
            print(' > Manually inputing data')
            print('TBD...')
            exit(1)
    ###########################################################################
    # Get advanced calculation parameters
    if (isfile(filename_adv)):
        # Read parameters from file
        print(' > Reading advanced parameters from input file')
        adv_params = read_input_adv(filename_adv,protocol)
    else:
        # Use default values
        print(' > Loading default advanced parameters')
        adv_params = get_input_adv(protocol)
        
    ###########################################################################
    # Check the values of the parameters
    print(' > Checking input parameters')
    if protocol.lower() == 'aBB84-WCP'.lower():
        from .protocol_inputs.input_efficient_BB84 import check_params
        main_params, adv_params = check_params(main_params,adv_params) # Dictionary of parameters
    else:
        raise ValueError('Protocol not recognised: {}'.format(protocol)) 
    return protocol, main_params, adv_params