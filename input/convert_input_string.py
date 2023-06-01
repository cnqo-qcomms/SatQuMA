# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 18:02:12 2021

@author: Duncan McArthur
"""

__all__ = ['read_protocol_param','list_str_to_float','list_str_to_int',
           'tuple_str_to_float','tuple_str_to_int']

###############################################################################

def read_protocol_param(param_string,tOptimise):
    """
    Determine values for a protocol parameter from an input string.

    Parameters
    ----------
    param_string : string, array-like
        Parameter value string(s), list or singular.
    tOptimise: boolean
        Optimise protocol parameters?

    Raises
    ------
    ValueError
        If param_string is a list it must have 2 or 3 values.
    TypeError
        The param_string must be of type 'list' or 'str'.

    Returns
    -------
    bool/float, list
        List of values for this protocol parameter.

    """
    
    if tOptimise:
        if type(param_string) == list:
            # Input is a list of strings
            tOptimise = True                   # Optimise this parameter
            if len(param_string) == 3:
                # Input list contains three values
                tInit = True                   # Initial value is specified
                lb    = float(param_string[0]) # Specify lower bound
                ub    = float(param_string[1]) # Specify upper bound
                val   = float(param_string[2]) # Initial value of parameter
            elif len(param_string) == 2:
                # Input list contains two values
                tInit = False                  # Initial value is not specified
                lb    = float(param_string[0]) # Specify lower bound
                ub    = float(param_string[1]) # Specify upper bound
                val   = None                   # Parameter is to be randomly intialised
            else:
                raise ValueError('len = {}'.format(len(param_string)))
        else:
            raise TypeError('Input should be a list, not a {}'.format(type(param_string)))
    else:
        if type(param_string) == str:
            # Input is a single string
            tOptimise = False                  # Do not optimise this parameter
            tInit     = True                   # Initialisation using val
            val       = float(param_string)    # Parameter value
            lb        = None                   # No bounds required
            ub        = None
        else:
            # Input is neither list nor string
            raise TypeError('type = {} {}'.format(param_string,type(param_string)))
    return list([tOptimise,tInit,val,lb,ub])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def list_str_to_float(list_str):
    """
    Take string value(s) representing float(s) and return as a list of values.

    Parameters
    ----------
    list_str : string, list-like
        List of strings, or single string.

    Returns
    -------
    list_float : float, list
        List of converted float values.

    """
    list_float = []
    if type(list_str) == list:
        for x in list_str:
            list_float.append(float(x))
    elif type(list_str) == str:
        list_float.append(float(list_str))
    else:
        print(" > Warning. Unexpected type passed to 'list_str_to_float()':",type(list_str))
    return list_float

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def list_str_to_int(list_str):
    """
    Take string value(s) representing integer(s) and return as a list of values.

    Parameters
    ----------
    list_str : string, list-like
        List of strings, or single string.

    Returns
    -------
    list_int : integer, list
        List of converted integer values.

    """
    list_int = []
    if type(list_str) == list:
        for x in list_str:
            list_int.append(int(x))
    elif type(list_str) == str:
        list_int.append(int(list_str))
    else:
        print(" > Warning. Unexpected type passed to 'list_str_to_int()':",type(list_str))
    return list_int

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def tuple_str_to_float(list_str):
    """
    Take string value(s) representing float(s) and return as a tuple of values.

    Parameters
    ----------
    list_str : string, list-like
        List of strings, or single string.

    Returns
    -------
    float, tuple
        Tuple of converted float values.

    """
    list_float = []
    if type(list_str) == list:
        for x in list_str:
            list_float.append(float(x))
    elif type(list_str) == str:
        list_float.append(float(list_str))
    else:
        print(" > Warning. Unexpected type passed to 'list_str_to_float()':",type(list_str))
    return tuple(*list_float)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def tuple_str_to_int(list_str):
    """
    Take string value(s) representing integer(s) and return as a tuple of values.

    Parameters
    ----------
    list_str : string, list-like
        List of strings, or single string.

    Returns
    -------
    integer, tuple
        Tuple of converted integer values.

    """
    list_int = []
    if type(list_str) == list:
        for x in list_str:
            list_int.append(int(x))
    elif type(list_str) == str:
        list_int.append(int(list_str))
    else:
        print(" > Warning. Unexpected type passed to 'list_str_to_int()':",type(list_str))
    return tuple(*list_int)