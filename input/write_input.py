# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 17:21:18 2021

@author: Duncan McArthur
"""

import numpy as np

__all__ = ['print_header', 'print_input']

###############################################################################

def _print_centre(string,width=80):
    """
    Print a centre justified string to StdOut.

    Parameters
    ----------
    string : str
        String to print.
    width : int, optional
        Line width for printing. The default is 80.

    Returns
    -------
    None.

    """
    print(' '*((max(width,len(string)) - len(string))//2) + string)
    return None
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def print_header():
    """
    Print the SatQuMA development header to StdOut.

    Returns
    -------
    None.

    """
    print('_'*80+'\n')
    symb = '*'
    n    = 5  # Number of '+' symbols
    ns   = int((80 - n)/(n + 1)) # Symbol spacing
    _print_centre((symb + ' '*ns)*(n - 1) + symb,80)
    print()
    strings = ['SatQuMA: Satellite Quantum Modelling & Analysis',
               'v2.0.0-beta',
               'D. McArthur, J. S. Sidhu, T. Brougham, R. G.-Pousa, and D. K. L. Oi',
               'University of Strathclyde',
               '01/05/2023',
               'https://github.com/cnqo-qcomms/SatQuMA']
    for string in strings:
        _print_centre(string,80)
    print('\n' + '_'*80 + '\n')
    return None

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def print_input(params_dict,strID):
    """
    Print the input parameters from a dictionary.

    Parameters
    ----------
    params_dict : dict
        Input parameters.
    strID : str
        Identifier to print along with parameters.

    Returns
    -------
    None.

    """
    
    if type(params_dict) is not dict:
        TypeError('{} is {}'.format(strID,type(params_dict)))
    
    for key0, value0 in params_dict.items():
        if isinstance(value0, dict):
            for key1, value1 in value0.items():
                if isinstance(value1,list) or isinstance(value1, tuple) or\
                    isinstance(value1,np.ndarray):
                    print('{}[{}][{}]:'.format(strID,key0,key1),*value1)
                else:
                    print('{}[{}][{}]:'.format(strID,key0,key1),value1)
        elif isinstance(value0, list) or isinstance(value0, tuple)  or\
            isinstance(value1,np.ndarray):
            print('{}[{}]:'.format(strID,key0),*value0)
        else:
            print('{}[{}]:'.format(strID,key0),value0)
