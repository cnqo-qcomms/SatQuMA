# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:04:32 2021

@author: Duncan McArthur
"""

__all__ = ['str2bool','uncomment_input','strip_quote_input','split_input',
           'no_whitespace','str_is_None','input_from_list']

###############################################################################

def str2bool(s):
    """
    Takes a string and evaluates if it belongs to a list of 'true' values:
        "yes", "y", "true", "t", "1",
    or a list of 'false' values:
        "no", "n", "false", "f", "0".
    Otherwise it is None.

    Parameters
    ----------
    s : string
        String to convert to boolean.

    Returns
    -------
    boolean
        String true value.

    """
    if str(s).lower() in ("yes", "y", "true", "t", "1"):
        return True
    elif str(s).lower() in ("no", "n", "false", "f", "0"):
        return False
    else:
        return None
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def uncomment_input(x,delim):
    """
    Strips newline characters and comments which appear after the specified
    delimiter from the input string x.

    Parameters
    ----------
    x : string
        String with delimited comments.
    delim : string
        Delimiter prefixing comments.

    Returns
    -------
    string
        String without delimited comments.

    """
    return x.split(delim)[0].strip()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def strip_quote_input(x):
    """
    Strip quotations from an input string.

    Parameters
    ----------
    x : string
        Input string with possible quotation marks.

    Returns
    -------
    string
        Input string without quotation marks.

    """
    return str(x).strip("'").strip('"')

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def split_input(x,delim):
    """
    Separate text input according to a specified delimiter.

    Parameters
    ----------
    x : string
        Input string.
    delim : string
        Input parameter separation character(s).

    Returns
    -------
    string, list-like
        Separated input string.

    """
    return x.split(delim)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def no_whitespace(x):
    """
    Removes whitespace from an input string.

    Parameters
    ----------
    x : string
        Input string.

    Returns
    -------
    string
        Output string sans whitespace.

    """
    return x.replace(" ", "")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def str_is_None(s):
    """
    Check if input string is 'None'

    Parameters
    ----------
    s : string
        Input string.

    Returns
    -------
    bool
        Is string equivalent to 'None'?

    """
    if no_whitespace(s).lower() == 'none':
        return True
    else:
        return False

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def input_from_list(data,thisLine,nLines,count,list_params,strParam='Input parameter'):
    """
    Read input string from data and check if it belongs to a list of accepted 
    strings. The data list may contain non-parameter values and so a count of
    parameters encountered is updated along with the last data index checked. 

    Parameters
    ----------
    data : string, array-like
        List of input strings.
    thisLine : integer
        List index to start from.
    nLines : integer
        No. of entries in list.
    count : integer
        Counter for parameters.
    list_params : string, array-like
        List of accepted strings.
    strParam : string, optional
        Parameter descriptor. The default is 'Input parameter'.

    Returns
    -------
    x : string
        Accepted parameter.
    thisLine : integer
        Updated index to continue from.
    count : integer
        Updated count of parameters.

    """
    while True and (thisLine < nLines):
        thisLine += 1
        x = uncomment_input(data[thisLine],'#') # Remove comments - anything following a '#'
        if x:
            count += 1
            # Data string is not empty
            x = strip_quote_input(no_whitespace(x)) # Remove white space
            break
    if strip_quote_input(x).lower() in (p.lower() for p in list_params):
        #print(strParam + ':',x)
        return x, thisLine, count
    else:
        print('Error!',strParam,'not recognised:',x)
        print("Use:",*["'{0}'".format(p) for p in list_params])
        raise ValueError('{} = {}'.format(strParam,x))
        #return None, thisLine, count