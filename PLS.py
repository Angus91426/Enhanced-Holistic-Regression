from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import numpy as np

def Partial_Least_Squares( X = None, y = None, ncomp = None ):
    pls = PLSRegression( n_components = ncomp )
    pls.fit( X, y )
    y_pred = pls.predict( X )
    coef_ = pls.coef_
    return coef_[0], y_pred

def bPLS( X0, Y0, ncomp ):
    """
    """
    
    n, p = X0.shape
    
    # Decomposition, construct loading, weight, and score matrices
    W = np.zeros( ( p, ncomp ) )
    T = np.zeros( ( n, ncomp ) )
    P = np.zeros( ( p, ncomp ) )
    
    w = X0.T @ Y0 / np.linalg.norm( X0.T @ Y0 )
    for a in range( ncomp ):
        t = X0 @ w
        t_norm = t / np.dot( t.T, t )
        
        # Loading vector
        p = X0.T @ t_norm
        
        # Update W, T, and P
        W[:, a] = w.flatten()
        T[:, a] = t_norm.flatten()
        P[:, a] = p.flatten()
        
        # Deflate X0
        X0 = X0 - np.dot( t.T, t ) * np.dot( t_norm, p.T )
        
        w = X0.T @ Y0
    
    # Zero replacement
    for a in range( ncomp - 1, -1, -1 ):
        coef = W @ T.T @ Y0
        if any( coef < 0 ):
            negative_idx = np.where( coef < 0 )[0]
            for idx in negative_idx:
                W[idx, a] = max( 0, W[idx, a] )
    
    # Calculate the final coefficient
    coef = ( W @ T.T @ Y0 ).flatten()
    return coef

def get_order(arr):
    # Pair each element with its index
    indexed_arr = list(enumerate(arr))
    # Sort the array based on the values
    sorted_arr = sorted(indexed_arr, key=lambda x: x[1])
    # Initialize the order list with zeros
    order = [0] * len(arr)
    # Assign ranks, handling duplicates by keeping the same rank for same values
    rank = 1
    for i, (index, value) in enumerate(sorted_arr):
        if i > 0 and value == sorted_arr[i - 1][1]:
            order[index] = order[sorted_arr[i - 1][0]]
        else:
            order[index] = rank
        rank += 1
    return order

def bPLS_CV( X, y, ncomp, normalize = True ):
    # Normalization
    if normalize:
        std = StandardScaler()
        X_normalized = std.fit_transform( X )
        y_normalized = std.fit_transform( y.reshape( -1, 1 ) )
    else:
        X_normalized = X
        y_normalized = y.reshape( -1, 1 )
    
    # bPLS
    bPLS_Best = {
        'ncomp': None,
        'R2': None,
        'Coef': None,
        'QSIFS': 0
    }
    for LV in range( 1, ncomp + 1 ):
        coef = bPLS( X_normalized, y_normalized, ncomp = LV )
        r2 = r2_score( y_normalized, X_normalized @ coef )
        
        nMOSFET_coef = coef[0:3]
        pMOSFET_coef = coef[3:]
        weight = [3, 2, 1]
        
        nMOSFET_order = get_order( nMOSFET_coef )
        pMOSFET_order = get_order( pMOSFET_coef )
        
        QSIFS = np.dot( nMOSFET_order, weight ) + np.dot( pMOSFET_order, weight ) + r2
        if QSIFS > bPLS_Best['QSIFS']:
            bPLS_Best['ncomp'] = LV
            bPLS_Best['R2'] = r2
            bPLS_Best['Coef'] = coef
            bPLS_Best['QSIFS'] = QSIFS
    return bPLS_Best