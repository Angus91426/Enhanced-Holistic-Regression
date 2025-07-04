import warnings
warnings.filterwarnings( 'ignore' )

import time
import pandas as pd, numpy as np, gurobipy as gp
import Python.Sparse_Holistic as Sparse_Holistic

import math, os, random

from gurobipy import GRB
from rich.text import Text
from rich import print
# from julia import Main

def Simulation_Data( n: int, p: int, MR: list, noise_scale: float, rand_seed: int, \
                    coef_min: int = 0, normalize: bool = False, transform_X: bool = False, original_X: np.array = None ) -> tuple[dict, dict, np.array]:
    """
    Generate simulation data for multicollinearity.
    
    Parameters
    ----------
    n : int
        Number of samples.
    p : int
        Number of features.
    MR : list
        List of the number of each kind of multicollinearity relationship.
    noise_scale : float
        Standard deviation of the Gaussian noise added to the multicollinear relationships.
    rand_seed : int
        Random seed.
    ***************************************************************
    coef_min: int, optional
        Minimum coefficient value. The default is 1.
    normalize : bool, optional
        Whether to normalize the design matrix. The default is False.
    
    Returns
    ----------
    Multicollinear_indices : dict
        The indices of the true multicollinearity relationships.
    
    Multicollinear_gammas : dict
        The coefficients of the true multicollinearity relationships.
    
    X : np.array
        The design matrix with the multi-collinear relationships.
    """
    
    random.seed( int( rand_seed ) )
    rng = np.random.default_rng( seed = rand_seed )
    
    # Generate initial X
    if transform_X:
        X = original_X
    else:
        X = rng.normal( loc = 0.0, scale = 1.0, size = ( n, p + 1 ) )
    
    noise = rng.normal( loc = 0.0, scale = noise_scale, size = ( n, p + 1 ) )
    
    # Total number of features in the multicollinearity relationships
    MR2, MR3, MR4, MR8, MR10 = MR
    no_var = 2 * MR2 + 3 * MR3 + 4 * MR4 + 8 * MR8 + 10 * MR10
    
    # Multicollinear relationship coefficients
    gamma = rng.uniform( -10, 10, size = no_var ).tolist()
    # Replace the gamma values that are in [-coef_min, coef_min]
    if coef_min != 0:
        unsatisfied_index = np.where( np.abs( gamma ) < coef_min )[0]
        
        while len( unsatisfied_index ) != 0:
            for index in unsatisfied_index:
                gamma[index] = rng.uniform( -10, 10 )
            unsatisfied_index = np.where( np.abs( gamma ) < coef_min )[0]
    
    # Multicollinear relationship indices
    col = random.sample( [i for i in range( 1, p + 1 )], no_var )
    
    # Save the indices and coefficients of the multicollinearity relationships into dict for returning
    indices, coef = {}, {}
    index = 0
    for num, relationships, name in zip( [2, 3, 4, 8, 10], MR, ['MR2', 'MR3', 'MR4', 'MR8', 'MR10'] ):
        indices[name] = []
        coef[name] = []
        for _ in range( relationships ):
            indices[name].append( col[index:index + num] )
            coef[name].append( gamma[index:index + num] )
            index += num
    
    # Multicollinear relationships
    for name in ['MR2', 'MR3', 'MR4', 'MR8', 'MR10']:
        for i in range( len( indices[name] ) ):
            X[:, indices[name][i][0]] = np.dot( X[:, indices[name][i][1:]], coef[name][i][1:] )
    
    # Add noise into the whole data
    X = X + noise
    
    # Transfer the first column into interception term
    X[:, 0] = 1
    
    # Normalize X
    if normalize:
        X, _, _ = Sparse_Holistic.Normalize( X_train = X, intercept = True )
    
    feature_col = np.array( ['Intercept'] + [f'X{i + 1}' for i in range( p )] )
    return indices, coef, X, feature_col

def Multicollinear_score( z_pos: list, sets: dict, verbose: bool = False ) -> tuple[float, float]:
    """
    Compute the accuracy and false positive rate of the detecting result for simulation data.
    
    Parameters
    ----------
    z_pos : list
        The list of the detected multicollinearity relationships.
    sets : dict
        The dictionary of the true multicollinearity relationships.
    
    Returns
    ----------
    ACC : float
        The percentage of finding the correct number of multicollinearity relationships.
    FPR : float
        The percentage of finding incorrect multicollinearity relationships.
    
    Source
    ----------
    Bertsimas, D., & Li, M. L. (2020). Scalable holistic linear regression. Operations Research Letters, 48(3), 203-208.
    Page 207.
    """
    
    total_set = 0
    for _, value in sets.items():
        total_set += len( value )
    
    # Accuracy
    if len( z_pos ) > total_set:
        ACC = 100
    else:
        ACC = ( len( z_pos ) / total_set ) * 100
    
    # False positive rate
    correct = 0
    for z in z_pos:
        if f'MR{len( z )}' not in sets.keys():
            incorrect = True
        else:
            incorrect = False
            sorted_z = sorted( z )
            for value in sets[f'MR{len( z )}']:
                sorted_set = sorted( value )
                if sorted_z == sorted_set:
                    correct += 1
                    incorrect = False
                    break
                else:
                    incorrect = True
        
        if incorrect:
            if verbose:
                print( f'z_pos: {z}, incorrect.' )
    
    FPR = ( 1 - ( correct / len( z_pos ) ) ) * 100
    return ACC, FPR

def small_eigvec( X: np.array, epsilon: float = None ) -> tuple[int, np.array]:
    """
    Generate the eigenvectors with eigenvalue smaller than epsilon.
    
    Parameters
    ----------
    X : np.array
        The data matrix.
    ***************************************************************
    epsilon : float, optional
        Threshold for eigenvalues. The default is None.
    
    Returns
    ----------
    small_eig_val : int
        The number of eigenvectors with eigenvalue smaller than epsilon.
    E : np.array
        The eigenvectors with eigenvalue smaller than epsilon.
    """
    
    # Compute eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eigh( X.T.dot( X ) )
    
    # Value of epsilon is Data-driven
    if epsilon is None:
        if Sparse_Holistic.Check_Normalization( X ):
            if X.shape[1] <= 810:
                epsilon = 10 ** (-2)
            elif X.shape[1] > 810 and X.shape[1] <= 940:
                epsilon = 10 ** (-3)
            elif X.shape[1] > 940 and X.shape[1] <= 980:
                epsilon = 10 ** (-4)
            else:
                epsilon = 10 ** (-5)
        else:
            if X.shape[1] >= 970:
                epsilon = 10 ** (-2)
            elif X.shape[1] >= 940 and X.shape[1] < 970:
                epsilon = 10 ** (-1)
            else:
                epsilon = 1
    
    small_eig_val = np.sum( eig_val < epsilon ) # number of the eigenvectors with eigenvalue < epsilon
    E = np.empty( ( X.shape[1], small_eig_val ) ) # Empty matrix for the eigenvectors with eigenvalue < epsilon
    index = 0
    for i in range( X.shape[1] ):
        if eig_val[i] < epsilon:  # If the eigenvalue is small.
            E[:, index] =  eig_vec[:, i] # plug in the eigenvector into V
            index += 1
    return small_eig_val, E

def min_support( V: np.array, z_pos: list, delta: float = 10 ** ( -4 ), time: float = 60, verbose: bool = False ) -> list:
    """
    Find the multicollinearity relationships by solving minimum support problem.
    
    Parameters
    ----------
    V : np.array
        The eigenvectors with eigenvalue smaller than epsilon.
    z_pos : list
        The list of the detected multicollinearity relationships.
    ***************************************************************
    delta : float, optional
        A positive constant that ensures that a ̸= 0. The default is 10 ** ( -4 ).
    time : float, optional
        Time limit for the solver. The default is None.
    
    Returns
    ----------
    detected_z : list
        The list of the detected multicollinearity relationship index.
    
    Source:
    ----------
    Bertsimas, D., & Li, M. L. (2020). Scalable holistic linear regression. Operations Research Letters, 48(3), 203-208.
    Page 206, problem (11).
    """
    
    try:
        with gp.Env( empty = True ) as env:
            env.setParam( 'OutputFlag', 0 )
            env.start()
            with gp.Model( "Holistic_regression", env = env ) as model:
                # model.Params.FeasibilityTol = 10**(-9)
                model.Params.IntFeasTol = 10**(-9)
                
                p, m = V.shape # m: the number of small eigenvalue
                
                M = 1 / math.sqrt( m )
                theta = model.addVars( m, vtype = GRB.CONTINUOUS, lb = -M, ub = M, name = "theta" ) 
                theta_abs = model.addVars( m, vtype = GRB.CONTINUOUS, lb = 0, name = "theta_abs" )
                z = model.addVars( p, vtype = GRB.BINARY, name = "z" )
                a = model.addVars( p, vtype = GRB.CONTINUOUS, lb = -M, ub = M, name = "a" )
                model.update()
                model.setObjective( z.sum(), GRB.MINIMIZE )
                model.addConstr( z.sum() >= 2 )
                
                for i in range( p ):
                    model.addConstr( a[i] == gp.quicksum( V[i, j] * theta[j] for j in range(m))  )
                
                # Add constraints by using z_pos
                if len( z_pos ) >= 1: # for example [[0, 1]],  [[0, 1], [2, 3, 4]]
                    for i in range( len( z_pos ) ):
                        model.addConstr(  gp.quicksum( z[ z_pos[i][j] ]  for j in range( len( z_pos[i] ) ) ) <= len( z_pos[i] ) - 1 )
                
                for i in range( p ):
                    model.addConstr( a[i] <= M * z[i] )
                    model.addConstr( a[i] >= -M * z[i] )
                
                for j in range( m ): 
                    model.addGenConstrAbs( theta_abs[j], theta[j] )
                
                model.addConstr( theta_abs.sum() >= delta )
                
                if time is not None:
                    model.Params.TimeLimit = time
                
                model.optimize()
                
                if model.status == GRB.INFEASIBLE:
                    return None
                else:
                    z_sol = np.array( [ z[i].x for i in range( p ) ] )
                    # print( Text( f'z.sum() = {z_sol.sum()}', style = 'orange') )
                    detected_z = np.where( z_sol == 1 )[0].tolist()
                    return detected_z
    
    except gp.GurobiError as e:
        print( "Error code" + str( e.errno ) + ": " + str( e ) )
        return False
    
    except AttributeError:
        print( "Encountered an attribute error" )
        model.computeIIS()
        model.write( "IISFile/Multicollienar/min_support.ilp" )
        return False

def Detection( X: np.array, detected_z: list = [], verbose: bool = False ) -> tuple[list, bool]:
    """
    Detect for multicollinearity relationships in the data X.
    
    Parameters
    ----------
    X : np.array
        The data matrix.
    ***************************************************************
    detected_z : list, optional
        The list of the detected multicollinearity relationships. The default is [].
    verbose : bool, optional
        Whether to print the number of detected relationships. The default is False.
    
    Returns
    ----------
    z_pos: list
        The list of the detected multicollinearity relationships.
    time_limit: bool
        Whether the time limit is reached.
    
    Source:
    ----------
    Bertsimas, D., & Li, M. L. (2020). Scalable holistic linear regression. Operations Research Letters, 48(3), 203-208.
    Page 206, Algorithm 1.
    """
    
    # Decide the number of collinear relationships to be detected
    _, V = small_eigvec( X )
    if verbose: print( f'Number of smaller eigenvalues: {V.shape[1]}' )
    
    z_pos = []
    if len( detected_z ) == 0:
        num_of_relationship = V.shape[1] # If no detected_z (relationships detected from reduced matrix), then detect all relationships
    else:
        num_of_relationship = V.shape[1] - len( detected_z )
    
    while len( z_pos ) < num_of_relationship:
        z_sol = min_support( V = V, z_pos = detected_z )
        if z_sol is None: # No relationship detected.
            print( 'End iteration because minimum support model is infeasible.' )
            break
        else:
            # print( f'Found z: {z_sol}' )
            detected_z.append( z_sol )
            z_pos.append( z_sol )
    
    return z_pos

def Inequality_Inspection( X: np.array, feature_idx: list, add_intercept: bool = False ) -> tuple[bool, bool, float]:
    """
    Solving the least square problem to check if there exist a vector a that satisfies the inequality.
    
    Parameters
    ----------
    X : np.array
        The data matrix.
    feature : list
        The indices of the detected multicollinearity relationships.
    ***************************************************************
    add_intercept : bool, optional
        Whether to add an intercept term. The default is False.
    
    Returns
    ----------
    inequality_satisfied: bool
        Whether the inequality is satisfied.
    Reach_Time_Limit: bool
        Whether the time limit is reached.
    norm: float
        The norm of objective function.
    
    Source:
    ----------
    Bertsimas, D., & Li, M. L. (2020). Scalable holistic linear regression. Operations Research Letters, 48(3), 203-208.
    Page 205, definition 1, problem (9).
    """
    with gp.Env( empty = True ) as env:
        env.setParam( 'OutputFlag', 0 )
        env.start()
        with gp.Model( "Holistic_regression", env = env ) as model:
            model.Params.NonConvex = 2
            model.Params.TimeLimit = 60
            epsilon = 10 ** (-2)
            
            if add_intercept:
                # Add a column of ones to X as the intercept
                X = np.column_stack( ( np.ones( X.shape[0] ), X ) )
            
            # Check if the first column is all ones
            if np.all( X[:, 0] == 1 ):
                X, _, _ = Sparse_Holistic.Normalize( X_train = X, intercept = True )
            else:
                X, _, _ = Sparse_Holistic.Normalize( X_train = X, intercept = False )
            
            partial_X = X[:, feature_idx]
            n, p = partial_X.shape
            
            A = model.addVars( p, vtype = gp.GRB.CONTINUOUS, lb = -10, ub = 10, name = 'A' )
            
            # 2-norm constraint for coefficient: ||A||_2 = 1
            model.addConstr( gp.quicksum( A[j] ** 2 for j in range( p ) ) == 1, name = 'norm_constr' )
            
            model.setObjective( gp.quicksum( ( gp.quicksum( partial_X[i, j] * A[j] for j in range( p ) ) * \
                                                gp.quicksum( partial_X[i, j] * A[j] for j in range( p ) ) ) for i in range( n ) ), gp.GRB.MINIMIZE )
            
            model.update()
            
            model.optimize()
            
            inequality_satisfied, infeasible, norm, A_opt = False, False, np.inf, None
            if model.status == gp.GRB.OPTIMAL:
                norm = np.sqrt( np.abs( model.objVal ) )
                inequality_satisfied = False if norm > epsilon else True
                A_opt = [A[j].x for j in range( p )] # Receive the optimal solution of the vector A
            else:
                infeasible = True
                if model.status == gp.GRB.TIME_LIMIT:
                    print( 'Time limit reached.' )
                else:
                    print( "Model infeasible for inequality inspection." )
            return inequality_satisfied, infeasible, norm, A_opt

def Dimensionality_Reduction( X: np.array, outlier_threshold: int, interception: bool = True ) -> tuple[np.ndarray, np.ndarray]:
    reduction_idx = [] # List of the index of the reduction result.
    
    # Create correlation coefficient matrix
    if interception:
        X_corr = pd.DataFrame( X[:, 1:] ).corr()
    else:
        X_corr = pd.DataFrame( X ).corr()
    
    # Sort the correlation coefficient in descending order for each column by absolute value
    df = {}
    for col in X_corr.columns:
        individual_corr = X_corr[col].reindex( X_corr[col].abs().sort_values( ascending = False ).index )
        
        df[f'{col}'] = individual_corr.index[1:]
        df[f'{col}_corr_coef'] = np.round( individual_corr.values[1:], 4 )
        df[f'{col}_normalized'] = np.round( ( individual_corr.values[1:] - individual_corr.values[1:].mean() ) / individual_corr.values[1:].std(), 4 )
    
    index = pd.DataFrame( df, columns = [f'{col}' for col in X_corr.columns] ).iloc[0, :].values
    normalize = pd.DataFrame( df, columns = [f'{col}_normalized' for col in X_corr.columns] ).iloc[0, :].values
    
    highest_corr = {}
    reduction_idx = []
    for col in X_corr.columns:
        temp_ = np.where( index == col )[0] 
        keep_idx = []
        for idx in temp_:
            if abs( normalize[idx] ) >= outlier_threshold:
                keep_idx.append( int( idx ) )
        
        if len( keep_idx ) > 1:
            highest_corr[col] = keep_idx
            reduction_idx += [col]
            reduction_idx += keep_idx
    
    if len( reduction_idx ) > 0:
        if interception:
            reduction_idx = [int( idx ) + 1 for idx in reduction_idx]
            reduction_idx.append( 0 )
        else:
            reduction_idx = [int( idx ) for idx in reduction_idx]
        reduction_idx = np.sort( reduction_idx )
        # Remove the duplicated index
        reduction_idx = np.unique( reduction_idx )
        reduction_X = X[:, reduction_idx]
    else:
        reduction_X, reduction_idx = None, None
    return reduction_X, reduction_idx

def Multicollinear_Detection(  X: np.array, outlier_threshold: int = None, col_names: np.ndarray = None, \
                                inspection: bool = True, reduction: bool = True, modified: bool = True, verbose: bool = False ):
    """
    Detect the multicollinearity relationships in the data matrix.
    
    Parameters
    ----------
    X : np.array
        The data matrix.
    col_names : np.ndarray
        The column names of the data matrix.
    inspection : bool, optional
        Whether to inspect the multicollinearity relationships.
    modified : bool, optional
        Whether to modify the multicollinearity relationships.
    verbose : bool, optional
        Whether to print the results.
    
    Returns
    ----------
    z_pos : list
        The list of multicollinearity relationships.
    reduce_p : int
        The number of features after dimensionality reduction.
    z_pos_original : list
        The list of multicollinearity relationships from the original matrix.
    z_pos_reduced : list
        The list of multicollinearity relationships from the reduced matrix.
    """
    
    _, p = X.shape
    # If Higher dimension, perform dimensionality reduction first and then detect the multicollinearity relationships.
    z_pos_reduced, z_pos_original = [], []
    norm_rec, A_rec = [], []
    reduce_p = None
    if p >= 100 and reduction:
        if outlier_threshold is None:
            outlier_threshold = 4
        
        # Perform dimensionality reduction process to receive the reduced matrix
        reduce_X, reduction_idx = Dimensionality_Reduction( X = X, outlier_threshold = outlier_threshold, interception = True )
        
        # Use the reduced matrix to detect the multicollinearity
        if reduce_X is not None:
            reduce_p = reduce_X.shape[1]
            reduced_detected = Detection( X = reduce_X, detected_z = [], verbose = verbose )
            
            if len( reduced_detected ) > 0:
                for z_sol in reduced_detected:
                    correct_idx = [int(_) for _ in reduction_idx[z_sol]] # Receive the correct index from the reduction idx list.
                    if inspection: # Perform inequality inspection process
                        inequality_satisfied, reach_Time_Limit, norm, A_opt = Inequality_Inspection( X = reduce_X, feature_idx = z_sol, add_intercept = False )
                        
                        if verbose:
                            if col_names is not None:
                                if inequality_satisfied:
                                    print( Text( f'Relationship {col_names[ correct_idx ]} detected and confirmed. ( norm = {norm}, A_opt = {A_opt} )', style = 'green' ) )
                                else:
                                    if reach_Time_Limit:
                                        print( Text( f'Relationship {col_names[ correct_idx ]} detected but failed confirmation. ( Cause: Time limit reached. )', style = 'red' ) )
                                    else:
                                        print( Text( f'Relationship {col_names[ correct_idx ]} detected but failed confirmation. ( Cause: {norm} > 0.01; A_opt = {A_opt} )', style = 'red' ) )
                            else:
                                if inequality_satisfied:
                                    print( Text( f'Relationship {correct_idx} detected and confirmed. ( norm = {norm}, A_opt = {A_opt} )', style = 'green' ) )
                                else:
                                    if reach_Time_Limit:
                                        print( Text( f'Relationship {correct_idx} detected but failed confirmation. ( Cause: Time limit reached. )', style = 'red' ) )
                                    else:
                                        print( Text( f'Relationship {correct_idx} detected but failed confirmation. ( Cause: {norm} > 0.01; A_opt = {A_opt} )', style = 'red' ) )
                        
                        if inequality_satisfied:
                            z_pos_reduced.append( correct_idx )
                            norm_rec.append( norm )
                            A_rec.append( A_opt )
                    else:
                        z_pos_reduced.append( correct_idx )
        
        if verbose:
            print( f'Detected {len( z_pos_reduced )} multicollinearity relationships from reduced matrix.' )
    else:
        if verbose:
            print( f'Notice: The number of features ( {p} ) is less than 100, no dimensionality reduction will be performed.' )
    
    if modified: # Modified multicollinearity detection, keep detecting by using the original matrix.
        # Compute the correct number of multicollinearity relationships by the original matrix X
        _, V = small_eigvec( X )
        correct_V_shape = V.shape[1]
        
        if len( z_pos_reduced ) < correct_V_shape:
            if verbose: print( f'Keeping finding {correct_V_shape - len( z_pos_reduced )} more multicollinearity relationships...' )
            
            # Using original matrix X to detect the rest of multicollinearity relationships
            original_detected = Detection( X = X, detected_z = z_pos_reduced.copy(), verbose = verbose )
            
            if len( original_detected ) > 0:
                for z_sol in original_detected:
                    if inspection:
                        inequality_satisfied, reach_Time_Limit, norm, A_opt = Inequality_Inspection( X = X, feature_idx = z_sol, add_intercept = False )
                        
                        if verbose:
                            if col_names is not None:
                                if inequality_satisfied:
                                    print( Text( f'Relationship {col_names[ z_sol ]} detected and confirmed. ( norm = {norm}, A_opt = {A_opt} )', style = 'green' ) )
                                else:
                                    if reach_Time_Limit:
                                        print( Text( f'Relationship {col_names[ z_sol ]} detected but failed confirmation. ( Cause: Time limit reached. )', style = 'red' ) )
                                    else:
                                        print( Text( f'Relationship {col_names[ z_sol ]} detected but failed confirmation. ( Cause: {norm} > 0.01; A_opt = {A_opt} )', style = 'red' ) )
                            else:
                                if inequality_satisfied:
                                    print( Text( f'Relationship {z_sol} detected and confirmed. ( norm = {norm}, A_opt = {A_opt} )', style = 'green' ) )
                                else:
                                    if reach_Time_Limit:
                                        print( Text( f'Relationship {z_sol} detected but failed confirmation. ( Cause: Time limit reached. )', style = 'red' ) )
                                    else:
                                        print( Text( f'Relationship {z_sol} detected but failed confirmation. ( Cause: {norm} > 0.01; A_opt = {A_opt} )', style = 'red' ) )
                        
                        if inequality_satisfied:
                            z_pos_original.append( z_sol )
                            norm_rec.append( norm )
                            A_rec.append( A_opt )
                    else:
                        z_pos_original.append( z_sol )
        z_pos = z_pos_reduced + z_pos_original
    else: # Normal multicollinearity detection, only using the reduced matrix.
        z_pos = z_pos_reduced.copy()
    
    if col_names is not None:
        involved_feature = []
        for z in z_pos:
            involved_feature.append( col_names[ z ] )
    else:
        involved_feature = None
    
    return z_pos, reduce_p, involved_feature, z_pos_original, z_pos_reduced, norm_rec, A_rec