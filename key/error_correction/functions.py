# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:15:35 2023

@author: Duncan McArthur
"""

import math
import numpy as np
from scipy.stats import binom

from ..maths import (h)

__all__ = ['logM','ErrorCorrect']

###############################################################################

def FInv(nX, QBERx, eps_c):
    """
    Calculates the quantile function, or inverse cumulative distribution
    function for a binomial distribution, nCk p^k (1 - p)^(n-k), where
    n = floor(nX), p = 1-QBERx, k \propto eps_c

    Parameters
    ----------
    nX : float/integer
        Number of events in sifted X basis.
    QBERx : float
        Quantum bit error rate in sifted X basis.
    eps_c : float
        Correctness error in the secure key.

    Returns
    -------
    float
        Quartile function of binomial distribution.

    """
    return binom.ppf(eps_c * (1. + 1.0 / np.sqrt(nX)), int(nX), 1.0 - QBERx)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def logM(nX, QBERx, eps_c):
    """
    This function os used to approximate the amount of secure key that is
    required to perform the error correction in bits.
    See Eq. (6) in [3].

    Parameters
    ----------
    nX : float
        The total number of measured events in the sifted X basis.
    QBERx : float
        The quantum bit error rate in the sifted X basis.
    eps_c : float
        The correctness error.

    Returns
    -------
    lM : float
        Number of bits for error correction.

    """
    lM = nX * h(QBERx) + (nX * (1.0 - QBERx) - FInv(int(nX), QBERx, eps_c) - \
                          1) * math.log((1.0 - QBERx) / QBERx) - \
        0.5*math.log(nX) - math.log(1.0 / eps_c)
    return lM

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def ErrorCorrect(val,fEC):
    """
    Calculates the error correction parameter \lambda_{EC}. Typical val is 1.16.
    Defined in Sec. IV of [1].

    Parameters
    ----------
    val : float
        Error correction factor.
    fEC : float
        Error correction efficiency.

    Returns
    -------
    float
        Error correction parameter.

    """
    return val * fEC