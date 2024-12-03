'''
S. Arman Ghaffarizadeh & Gerald J. Wang
Getting over the Hump with KAMEL-LOBE: Kernel-Averaging Method to Eliminate Length-Of-Bin Effects in Radial Distribution Functions
Journal of Chemical Physics (2023)
'''

import numpy as np
from scipy.stats import norm
import scipy.sparse as sp

''' 
python version of KAMEL_LOBE
Nick Hattrup
December 2, 2024
'''


def compute_T1(delr, Nbins):
    """
    Args:
        delr (float): bin width
        Nbins (int): number of bins

    Returns:
        T1 (array): T1 matrix of size Nbins x Nbins
    """
    T1 = np.zeros((Nbins,Nbins)) 
    for col in range(1, Nbins): # First column is all zeros
        value = (col*delr)**2
        T1[col, col] = value
        T1[col+1:, col] = 2*value
    return T1

def compute_T2(delr, Nbins, w):
    """
    Args:
        delr (float): bin width
        Nbins (int): number of bins

    Returns:
        T2 (array): T2 matrix of size Nbins x Nbins
    """
    m_KL = np.ceil(2*w/delr).astype(int) # number of bins to average over
    k_KL = 2*m_KL-1
    fractions = np.zeros((1,k_KL))
    A1_block = sp.identity(m_KL, format='csr')
    A2_block = sp.lil_matrix((m_KL, Nbins-m_KL))
    fractions[0,m_KL-1:] = norm.cdf(((np.arange(0,m_KL)+0.5)*delr),0,w)-norm.cdf(((np.arange(0,m_KL)-0.5)*delr),0,w)        
    fractions[0,:m_KL-1] = np.flip(fractions[0,m_KL:2*m_KL-1])
    fractions[0, :] *= 1/np.sum(fractions)
    B_block = sp.diags(np.tile(fractions, (Nbins-2*m_KL, 1)).T, np.arange(0, 2*(m_KL-1)+1), shape=(Nbins-2*m_KL, Nbins))
    T2 = sp.vstack((sp.hstack((A1_block, A2_block)), B_block, sp.hstack((A2_block, A1_block))))
    return T2

def compute_T3(delr, Nbins):
    """
    Args:
        delr (float): bin width
        Nbins (int): number of bins
        density (float): density of the system

    Returns:
        T3 (array): T3 matrix of size Nbins x Nbins
    """
    T3 = np.zeros((Nbins,Nbins))
    constant = 1/(delr**2)
    for row in range(1, Nbins):
        T3[row, row] = constant/(row)**2
        factor = 2*constant/(row)**2
        sign = 1 - 2 * (row & 1)
        for col in range(row):
            T3[row, col] = sign*factor
            sign *= -1
    # T3 /= 2.*np.pi*density*delr
    return T3


def KAMEL_LOBE(r,RDF,w=0.015):
    """
    Args:
        r (array): vector of equispaced radii at which RDF is evaluated
        RDF (array): vector of corresponding RDF values
        varargin (float, optional): width of Gaussian kernel (set to 0.015 by default)

    Returns:
        r_tilde (array): vector of equispaced radii at which KAMEL-LOBE RDF is evaluated
        gr_tilde (array): vector of corresponding KAMEL-LOBE RDF values
    """
    Nbins = RDF.shape[0] # number of bins
    delr = r[1]-r[0] # bin width, MATLAB version uses r[2]-r[1]
    m_KL = np.ceil(2*w/delr).astype(int) # number of bins to average over
    print(m_KL)

    if m_KL > 1:
        T1 = compute_T1(delr, Nbins)
        T2 = compute_T2(delr, Nbins, w)
        T3 = compute_T3(delr, Nbins)

        # Computing gr_tilde
        gr_convert = T3@(T2@(T1@RDF)) # Explicit operation order to reduce redundant matrix multiplications
        gr_tilde = gr_convert[:-m_KL]    
        r_tilde = r[:-m_KL]
        return r_tilde, gr_tilde
    
    else:
        gr_tilde = RDF
        r_tilde = r
        print('w <= delr/2, no averaging is performed')
        return r_tilde, gr_tilde

