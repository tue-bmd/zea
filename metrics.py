"""
==============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : metrics.py

    Author(s)     : Tristan Stevens
    Date          : 18 Nov 2021

==============================================================================
"""

import numpy as np

def CNR(x, y):
    """ Calculate contrast to noise ratio.

    Args:
        x: image
        y: image
        
    Returns:
        contrast to noise ratio.
    
    """
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    
    var_x = np.var(x)
    var_y = np.var(y)
    
    CNR = 20 * np.log10(np.abs(mu_x - mu_y) / np.sqrt((var_x + var_y) / 2))
    return CNR

