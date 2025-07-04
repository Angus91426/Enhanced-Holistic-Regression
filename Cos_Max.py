import warnings
warnings.filterwarnings( 'ignore' )

import pandas as pd, numpy as np
import Python.Sparse_Holistic as Sparse_Holistic

from itertools import chain

def Eigenvector_Loadings( X: np.array, col_names: list = None ) -> pd.DataFrame:
    # Normalize X to have zero mean and unit l2-norm
    X, _  = Sparse_Holistic.Normalize( X )
    
    R = np.dot( X.T, X )
    eig_val, eig_vec = np.linalg.eigh( R )
    if col_names is not None:
        eig_vec_df = pd.DataFrame( eig_vec.T, columns = col_names )
    else:
        eig_vec_df = pd.DataFrame( eig_vec.T, columns = [f'X{i + 1}' for i in range( X.shape[1] )] )
    df = pd.concat( [pd.DataFrame( eig_val, columns = ['Eigenvalues'] ), eig_vec_df], axis = 1 ).iloc[::-1]
    return df, eig_vec, eig_val

def Variance_Decomposition_Proportion( X: np.array, col_names: list = None ) -> np.array:
    # Normalize X to have zero mean and unit l2-norm
    X, _  = Sparse_Holistic.Normalize( X )
    
    R = np.dot( X.T, X )
    eig_val, eig_vec = np.linalg.eigh( R )
    # Computing VIF based on eigenvectors and eigenvalues
    VIF = np.zeros( X.shape[1] )
    for i in range( X.shape[1] ):
        VIF[i] = sum( ( ( eig_vec[i, j] ** 2) / eig_val[j] ) for j in range( X.shape[1] ) )
    
    # Computing VDP and Condition indices based on eigenvectors, eigenvalues, and VIF
    VDP = np.zeros( ( X.shape[1], X.shape[1] ) )
    condition_indices = np.zeros( X.shape[1] )
    for i in range( X.shape[1] ):
        condition_indices[i] = np.max( eig_val ) / eig_val[i]
        for j in range( X.shape[1] ):
            VDP[i, j] = ( ( eig_vec[i, j] ** 2) / eig_val[j] ) / VIF[i]
    
    if col_names is not None:
        df = pd.DataFrame( VDP.T, columns = col_names )
    else:
        df = pd.DataFrame( VDP.T, columns = [f'X{i + 1}' for i in range( X.shape[1] )] )
    
    df = pd.concat( [pd.DataFrame( condition_indices, columns = ['Condition_indices'] ), df], axis = 1 ).iloc[::-1]
    return df

def Total_Variation( X: np.array, col_names: list = None ) -> pd.DataFrame:
    # Add interception term into matrix X
    X = np.c_[ np.ones( X.shape[0] ), X ]
    
    if Sparse_Holistic.Check_Normalization( X ):
        X = X.copy()
    else:
        X, _ = Sparse_Holistic.Normalize( X )
    
    n, p = X.shape
    unit_vector = np.ones( n )
    # Compute total variation (TV) for each column of X
    TV = np.zeros( ( 1, p ) )
    for i in range( p ):
        TV[0, i] = np.dot( X[:, i].T, X[:, i] ) - ( n ** ( -1 ) ) * ( np.dot( unit_vector.T, X[:, i] ) ** 2 )
    
    if col_names is not None:
        df = pd.DataFrame( TV, columns = ['Intercept'] + col_names )
    else:
        df = pd.DataFrame( TV, columns = ['Intercept'] + [f'X{i + 1}' for i in range( X.shape[1] )] )
    return df

def Cos_Max( X: np.array ) -> np.array:
    """
    Using cos-max method to detect collinear relationships.
    
    Parameters
    ----------
    X : np.array
        The data matrix.
    
    Returns
    ----------
    A : np.array
        The transformation matrix.
    
    Source:
    ----------
    Shabuz, Z. R., & Garthwaite, P. H. (2024). Examining collinearities. Australian & New Zealand Journal of Statistics.
    """
    # Normalize X to have zero mean and unit l2-norm
    X, _, _  = Sparse_Holistic.Normalize( X_train = X, intercept = True )
    
    # Compute R^(-1/2) using eigenvalue decomposition
    R = np.dot( X.T, X )
    
    eig_val, eig_vec = np.linalg.eigh( R )
    # Avoid negative eigenvalues
    eig_val = np.abs( eig_val )
    
    A = eig_vec @ np.diag( 1 / np.sqrt( eig_val ) ) @ eig_vec.T
    return A

def Find_Identical( z_sol: dict ):
    '''
    Find identical list through the dictionary.
    '''
    identical = []
    for idx, list_ in z_sol.items():
        if len( list_ ) <= 2: continue
        count = 0
        for z in list_:
            if z_sol[z] == list_:
                count += 1
            else:
                break
        
        if count == len( list_ ) and list_ not in identical:
            identical.append( list_ )
    return identical

def Identify( detected_z ):
    length = {}
    for idx, list_ in detected_z.items():
        length[idx] = len( list_ )
    
    sorted_length = dict( sorted( length.items(), key = lambda x: x[1], reverse = False ) )
    z_sol = []
    for key, value in sorted_length.items():
        if value < 2: continue
        
        intersection = detected_z[key].copy()
        for z in detected_z[key]:
            intersection = list( set( intersection ) & set( detected_z[z] ) )
        
        if len( intersection ) > 1 and intersection not in z_sol:
            z_sol.append( intersection )
        
    return z_sol