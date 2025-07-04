import warnings
warnings.filterwarnings( 'ignore' )

import time, math, random

import numpy as np, pandas as pd, gurobipy as gp

from gurobipy import GRB, Model
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def Data_Preprocessing( X: pd.DataFrame, drop_first: bool = True, drop_single: bool = True, drop_perfect: bool = True ) -> ( pd.DataFrame, list ):
    
    # 1. Split into numerical and categorical features
    numerical_features = X.select_dtypes( exclude = ['object'] )
    num_col = numerical_features.columns
    categorical_features = X.select_dtypes( include = ['object'] )
    
    # 2. One-hot encode categorical features
    if not categorical_features.empty:
        encoded_categorical_df = pd.get_dummies( categorical_features, dtype = int, drop_first = drop_first )
        
        # 3. Remove encoded columns with only one '1'
        if drop_single:
            unique_cols = encoded_categorical_df.columns[encoded_categorical_df.sum( axis = 0 ) > 1]
            encoded_categorical_df = encoded_categorical_df[unique_cols]
    else:
        encoded_categorical_df = pd.DataFrame( columns = [] )
    
    # 4. Concat numerical and categorical features
    X = pd.concat( [numerical_features, encoded_categorical_df], axis = 1 )
    
    # 5. Drop constant features
    X = X.loc[:, ( X != X.iloc[0] ).any()]
    
    # 6. Implement Drop_Perfect for perfect collinearity
    if drop_perfect:
        X = Drop_Perfect( X )
    
    # 7. Add intercept term
    X = pd.concat( [pd.DataFrame( np.ones( ( X.shape[0], 1 ) ), columns = ['Intercept'] ), X], axis = 1 )
    
    # 8. Obtain column names for categorical features
    categorical_feature_names = [_ for _ in X.columns if _ not in num_col and _ != 'Intercept']
    
    # 9. Obtain the feature index for the group sparsity
    group_sparsity = []
    for cat_ in categorical_features.columns.tolist():
        temp_ = []
        for col_ in categorical_feature_names:
            if col_.startswith( cat_ ):
                temp_.append( X.columns.get_loc( col_ ) )
        
        if len( temp_ ) < 2: continue
        group_sparsity.append( temp_ )
    
    return X, categorical_feature_names, group_sparsity

def Special_Kfold( X: pd.DataFrame, cat_feat: list, n_splits: int = 5, shuffle: bool = True ):
    train_idx = []
    test_idx = []
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold( n_splits = n_splits, shuffle = shuffle )
    
    # We need to create a stratification label based on the categorical features
    stratify_labels = X[cat_feat].apply( lambda row: '_'.join( row.astype( str ) ), axis = 1 )
    
    for train_index, test_index in skf.split( X, stratify_labels ):
        train_idx.append( train_index )
        test_idx.append( test_index )
    
    return train_idx, test_idx

def Check_Normalization( data: np.ndarray, method: str = 'unit_length', intercept: bool = True ) -> bool:
    """
    Check if the data is normalized to desire method.
    
    Parameters
    ----------
    data : np.ndarray
        Data to be checked.
    ***************************************************************
    method : str, optional
        Normalization method. The default is 'unit_length'.
    intercept : bool, optional
        Whether the intercept is included in the data. The default is True.
    
    Returns
    ----------
    bool
        True if the data is normalized, False otherwise.
    """
    if method == 'unit_length':
        if len( data.shape ) == 2:
            # Skip the first column if intercept is True
            start_col = 1 if intercept else 0
            means = np.round( np.mean( data[:, start_col:], axis = 0 ), 3 )
            norms = np.round( np.linalg.norm( data[:, start_col:], axis = 0 ), 3 )
            if not ( np.all( means == 0 ) and np.all( norms == 1 ) ):
                return False
        else:
            mean = np.round( np.mean( data ), 3 )
            norm = np.round( np.linalg.norm( data ), 3 )
            if mean != 0 or norm != 1:
                return False
    elif method == 'Z-score':
        if len( data.shape ) == 2:
            # Skip the first column if intercept is True
            start_col = 1 if intercept else 0
            means = np.round( np.mean( data[:, start_col:], axis = 0 ), 3 )
            stds = np.round( np.std( data[:, start_col:], axis = 0 ), 3 )
            if not ( np.all( means == 0 ) and np.all( stds == 1 ) ):
                return False
        else:
            mean = np.round( np.mean( data ), 3 )
            std = np.round( np.std( data ), 3 )
            if mean != 0 or std != 1:
                return False
    return True

def Split_data( X: np.array, y: np.array, rand_seed = 426 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the dataset into 60%/20%/20% training/testing/validation.
    
    Parameters
    ----------
    X : np.array
        Feature matrix.
    y : np.array
        Target column.
    ***************************************************************
    rand_seed : int, optional
        Random seed for the data splitting. The default is 426.
    
    Returns
    ----------
    X_train : np.array
        Training feature matrix.
    y_train : np.array
        Training target column.
    X_test : np.array
        Testing feature matrix.
    y_test : np.array
        Testing target column.
    X_valid : np.array
        Validation feature matrix.
    y_valid : np.array
        Validation target column.
    
    Source
    ----------
    Bertsimas, Dimitris, and Michael Lingzhi Li. "Scalable holistic linear regression." 
    Operations Research Letters 48.3 (2020): 203-208. Page 207, section 5, paragraph 3.
    """
    X_train, X, y_train, y = train_test_split( X, y, test_size = 0.4, random_state = rand_seed ) # 0.6 for training, 0.4 for validation and testing
    X_test, X_valid, y_test, y_valid = train_test_split( X, y, test_size = 0.5, random_state = rand_seed ) # 0.5 for validation and 0.5 testing
    return X_train, y_train, X_test, y_test, X_valid, y_valid

def Separate_Normalize( X_train: np.array, X_valid: np.array = None, X_test: np.array = None, Categorical_Col: list = [] ):
    # Separate numerical and categorical features
    X_train_cat = X_train[Categorical_Col]
    X_train_num = X_train.drop( columns = Categorical_Col, axis = 1 )
    train_num_col = X_train_num.columns
    
    if X_valid is not None:
        X_valid_cat = X_valid[Categorical_Col]
        X_valid_num = X_valid.drop( columns = Categorical_Col, axis = 1 )
        valid_num_col = X_valid_num.columns
    if X_test is not None:
        X_test_cat = X_test[Categorical_Col]
        X_test_num = X_test.drop( columns = Categorical_Col, axis = 1 )
        test_num_col = X_test_num.columns
    
    # Only normalize the numerical features
    if X_valid is None and X_test is None:
        X_train_num, _, _ = Normalize( X_train_num.to_numpy(), intercept = True )
    elif X_valid is not None and X_test is None:
        X_train_num, X_valid_num, _ = Normalize( X_train_num.to_numpy(), X_valid_num.to_numpy(), intercept = True )
    elif X_valid is None and X_test is not None:
        X_train_num, X_test_num, _ = Normalize( X_train_num.to_numpy(), X_test_num.to_numpy(), intercept = True )
    else:
        X_train_num, X_valid_num, X_test_num, _ = Normalize( X_train_num.to_numpy(), X_valid_num.to_numpy(), X_test_num.to_numpy(), intercept = True )
    
    # Concat normalized numerical and categorical features
    X_train = pd.concat( [pd.DataFrame( X_train_num, columns = train_num_col ), X_train_cat], axis = 1 )
    
    if X_valid is not None:
        X_valid = pd.concat( [pd.DataFrame( X_valid_num, columns = valid_num_col ), X_valid_cat], axis = 1 )
    if X_test is not None:
        X_test = pd.concat( [pd.DataFrame( X_test_num, columns = test_num_col ), X_test_cat], axis = 1 )
    
    return X_train, X_valid, X_test

def Normalize( X_train: np.array, X_valid: np.array = None, X_test: np.array = None, intercept: bool = False ) -> tuple[np.array, np.array]:
    """
    Normalize the data to have zero mean and unit l2-norm.
    
    Parameters
    ----------
    X_train : np.array
        Training data to be normalized.
    ***************************************************************
    X_test : np.array, optional
        Testing data to be normalized. The default is None.
    intercept : bool, optional
        Whether the intercept is included in the data. The default is False.
    
    Returns
    ----------
    Normalized_train : np.array
        Normalized training data.
    Normalized_test: np.array
        Normalized testing data.
    """
    _, p = X_train.shape
    Normalized_train = np.ones( X_train.shape )
    train_data_mean = np.mean( X_train, axis = 0 )
    
    if X_valid is not None:
        Normalized_valid = np.ones( X_valid.shape )
    else:
        Normalized_valid = None
    
    if X_test is not None:
        Normalized_test = np.ones( X_test.shape )
    else:
        Normalized_test = None
    
    if intercept: # The interception term should not be normalized
        train_data_mean[0] = 0
        
        for i in range( 1, p ):
            Normalized_train[:, i] = ( X_train[:, i] - train_data_mean[i] ) / np.linalg.norm( X_train[:, i] - train_data_mean[i] )
            if X_test is not None:
                Normalized_test[:, i] = ( X_test[:, i] - train_data_mean[i] ) / np.linalg.norm( X_train[:, i] - train_data_mean[i] )
            if X_valid is not None:
                Normalized_valid[:, i] = ( X_valid[:, i] - train_data_mean[i] ) / np.linalg.norm( X_train[:, i] - train_data_mean[i] )
    else:
        for i in range( p ):
            Normalized_train[:, i] = ( X_train[:, i] - train_data_mean[i] ) / np.linalg.norm( X_train[:, i] - train_data_mean[i] )
            if X_test is not None:
                Normalized_test[:, i] = ( X_test[:, i] - train_data_mean[i] ) / np.linalg.norm( X_train[:, i] - train_data_mean[i] )
            if X_valid is not None:
                Normalized_valid[:, i] = ( X_valid[:, i] - train_data_mean[i] ) / np.linalg.norm( X_train[:, i] - train_data_mean[i] )
    return Normalized_train, Normalized_valid, Normalized_test

def Drop_Perfect( data: pd.DataFrame ) -> pd.DataFrame:
    """
    Drop one of the columns when their correlation coefficient is 1 or -1.
    
    Parameters
    ----------
    data : np.array
        Feature matrix.
    
    Returns
    ----------
    data : np.array
        Clean data.
    """
    
    # Compute the correlation coefficient matrix
    corr_coef = data.corr().round( 4 )
    
    # Receive the upper triangle elements of the correlation coefficient matrix (not including the diagonal)
    corr_coef = corr_coef.where( np.triu( np.ones_like( corr_coef ), 1 ) == 1 ).to_numpy()
    
    # Receive the index where the correlation coefficient is 1 or -1
    drop_index = np.where( np.abs( corr_coef ) == 1 )
    
    data = data.drop( columns = data.columns[drop_index[0]], axis = 1 )
    
    return data

def generate_initialization_of_beta( p: int, k: int, num_samples: int ) -> np.array:
    """
    Generate the initial solution for beta.
    
    Parameters
    ----------
    p : int
        Number of features.
    k : int
        The number of the nonzero elements in beta.
    num_samples : int
        Number of initial solution to generate.
    
    Returns
    ----------
    beta_list : np.array
        List of generated beta.
    
    Source:
    ----------
    Bertsimas, D., King, A., & Mazumder, R. (2016). Best subset selection via a modern optimization lens.
    Page 837, comment 6.
    """
    
    beta_list = np.zeros( ( num_samples, p ) )
    
    for i in range( num_samples ):
        epsilon = np.random.multivariate_normal( np.zeros( p ), 4 * np.eye( p ) )
        
        beta = np.minimum( i - 1, 1 ) * epsilon
        # Truncate beta to ensure the l0-norm is less than or equal to k
        nonzero_indices = np.nonzero( beta )[0]
        if len( nonzero_indices ) > k:
            remove_indices = np.random.choice( nonzero_indices, len( nonzero_indices ) - k, replace = False )
            beta[remove_indices] = 0
        
        beta_list[i] = beta
    return beta_list

def gradient_function( X: np.array, y: np.array, beta: np.array ) -> np.array:
    """
    Calculate the gradient of the loss function at beta.
    
    Parameters
    ----------
    X : np.array
        Feature matrix.
    y : np.array
        Target column.
    beta : np.array
        Beta used to calculate the gradient.
    
    Returns
    ----------
    gradient : np.array
        Gradient of the loss function at beta.
    
    Source:
    ----------
    Bertsimas, D., King, A., & Mazumder, R. (2016). Best subset selection via a modern optimization lens.
    Page 833, section 3.2.
    """
    gradient = -np.dot( X.T, ( y - np.dot( X, beta ) ) )
    return gradient

def loss_function( X: np.array, y: np.array, beta: np.array ) -> float:
    """
    Calculate the value of loss function at beta.
    
    Parameters
    ----------
    X : np.array
        Feature matrix.
    y : np.array
        Target column.
    beta : np.array
        Beta used to calculate the loss function.
    
    Returns
    ----------
    func_val : float
        Value of the loss function at beta.
    
    Source:
    ----------
    Bertsimas, D., King, A., & Mazumder, R. (2016). Best subset selection via a modern optimization lens.
    Page 833, section 3.2.
    """
    func_val = ( np.linalg.norm( y - np.dot( X, beta ) ) ** 2 ) / 2
    return func_val

def find_lambda_m( X: np.array, y: np.array, beta: np.array, eta: np.array, OutputFlag: bool = False ) -> float:
    """
    Solve the optimization problem to receive the value of lambda_m.
    
    Parameters
    ----------
    X : np.array
        Feature matrix.
    y : np.array
        Target column.
    beta : np.array
        The value of beta of the current iteration.
    eta : np.array
        The value of H_k( c ) of the current iteration.
    ***************************************************************
    OutputFlag : bool, optional
        Whether to show the solving process of Gurobi or not. The default is False.
    
    Returns
    ----------
    lambda_m_sol : float
        Optimal solution of the lambda_m.
    
    Source:
    ----------
    Bertsimas, D., King, A., & Mazumder, R. (2016). Best subset selection via a modern optimization lens.
    Page 833, Algorithm 2, step 2.
    """
    
    with gp.Env( empty = True ) as env:
        env.setParam( 'OutputFlag', 1 ) if OutputFlag else env.setParam( 'OutputFlag', 0 )
        env.start()
        with gp.Model( "find_lambda_m", env = env ) as model:
            try:
                bound = 50
                
                model.setParam( 'BarHomogeneous', 1  )
                
                error_lb = y.min() - 10 * ( abs( y.min() ) + abs( y.max() ) )
                error_ub = y.min() + 10 * ( abs( y.min() ) + abs( y.max() ) )
                
                lambda_m = model.addVar( vtype = GRB.CONTINUOUS, lb = 0, ub = 1, name = "lambda_m" )
                loss_beta = model.addVars( X.shape[1], vtype = GRB.CONTINUOUS, lb = -bound, ub = bound, name = "loss_beta" )
                error = model.addVars( X.shape[0], vtype = GRB.CONTINUOUS, lb = error_lb, ub = error_ub, name = "error" )
                
                model.addConstrs( loss_beta[i] == ( lambda_m * eta[i] ) + ( ( 1 - lambda_m ) * beta[i] ) for i in range( X.shape[1] ) )
                model.addConstrs( error[j] == y[j] - gp.quicksum( X[j, i]*loss_beta[i] for i in range( X.shape[1] ) ) for j in range( X.shape[0] ) )
                
                obj = gp.quicksum( error[j]**2 for j in range( X.shape[0] ) ) / 2
                model.setObjective( obj, GRB.MINIMIZE )
                model.optimize()
                
                lambda_m_sol = lambda_m.x
                
                return lambda_m_sol
            except AttributeError:
                print( "Encountered an attribute error while finding lambda_m" )
                model.computeIIS()
                model.write( "Out/lambda_m_IIS.ilp" )
                return None

def H_k( c_pool: np.array, k: int ) -> np.array:
    """
    Compute the value of H_k( c ).
    
    Parameters
    ----------
    c_pool : np.array
        The candidate set of c.
    k : int
        The maximum number of feature selected.
    
    Returns
    ----------
    sol : np.array
        The value of H_k( c ).
    
    Source:
    ----------
    Bertsimas, D., King, A., & Mazumder, R. (2016). Best subset selection via a modern optimization lens.
    Page 828, problem 3.3, formulation 3.4
    """
    # Sort the absolute values of c_pool in descending order
    sorted_indices = np.argsort( -np.abs( c_pool ) )
    
    # Select the first k indices from the sorted indices
    selected_indices = sorted_indices[:k]
    
    # Create a solution vector with zeros
    sol = np.zeros( len( c_pool ) )
    
    # Assign the corresponding values from c_pool to the solution vector
    sol[selected_indices] = c_pool[selected_indices]
    
    return sol

def Discrete_First_Order( initial_beta: np.array, k: int, X: np.array, y: np.array, tolerance: float, algorithm: int = 2, OutputFlag: bool = False ) -> tuple[np.array, float]:
    """
    Discrete First Order algorithm.
    
    Parameters
    ----------
    initial_beta : np.array
        The initial value of beta.
    k : int
        The maximum number of feature selected.
    X : np.array
        Feature matrix.
    y : np.array
        Target column.
    tolerance : float
        The tolerance of stopping the algorithm.
    ***************************************************************
    algorithm : int, optional
        Algorithm type. The default is 2.
    OutputFlag : bool, optional
        Whether to show the solving process of Gurobi or not. The default is False.
    
    Returns
    ----------
    optimal_beta_solution : np.array
        Optimal solution of beta.
    optimal_obj : float
        Optimal value of the loss function.
    
    Source:
    ----------
    Bertsimas, D., King, A., & Mazumder, R. (2016). Best subset selection via a modern optimization lens.
    Page 830, Algorithm 1; page 833, Algorithm 2
    """
    
    iterations = 1
    max_iterations = 100
    
    original_beta = initial_beta.copy()
    
    eigenvalues, _ = np.linalg.eigh( X.T.dot( X ) )
    max_eigen = max( eigenvalues ) 
    L = max( 1, round( 2 * max_eigen ) )
    
    optimal_beta_solution, optimal_obj = None, np.inf
    while iterations <= max_iterations:
        # Calculate the parameter c of H_k(c)
        c_pool = original_beta - ( gradient_function( X, y, original_beta ) / L )
        
        if algorithm == 1:
            new_beta = H_k( c_pool, k )
        else:
            # Find eta, the vector only keep the first k highest absolute value in c_pool
            eta = H_k( c_pool, k )
            
            # solve the optimize problem to receive lambda_m
            lambda_m = find_lambda_m( X, y, original_beta, eta, OutputFlag = OutputFlag )
            
            if lambda_m is None:
                optimal_beta_solution, optimal_obj = None, np.inf
                break
            
            new_beta = lambda_m * eta + ( 1 - lambda_m ) * original_beta
        
        # Calculate the value of loss function
        original_loss_function = loss_function( X, y, original_beta )
        new_loss_function = loss_function( X, y, new_beta )
        
        # Calculate the gap between the new and original value loss function
        loss_function_gap = abs( new_loss_function - original_loss_function )
        
        if loss_function_gap < tolerance:
            optimal_beta_solution = new_beta.copy()
            optimal_obj = new_loss_function
            break
        
        if new_loss_function < optimal_obj:
            optimal_beta_solution = new_beta.copy()
            optimal_obj = new_loss_function
        
        original_beta = new_beta
        iterations += 1
    
    return optimal_beta_solution, optimal_obj

def Warm_Start_MIO( X: np.array, y: np.array, k: int, initial_sol_size: int = 5 ) -> tuple[float, np.array]:
    """
    MIO with warm start.
    
    Parameters
    ----------
    X : np.array
        Feature matrix.
    y : np.array
        Target column.
    k : int
        The maximum number of feature selected.
    ***************************************************************
    initial_sol_size : int, optional
        The number of initial solutions. The default is 5.
    
    Returns
    ----------
    obj_val : float
        Optimal objective function value.
    hybrid_sol : np.array
        Optimal solution of beta.
    
    Source:
    ----------
    Bertsimas, D., King, A., & Mazumder, R. (2016). Best subset selection via a modern optimization lens.
    Page 827, section 2.3.3, for calculating specific bounds
    """
    _, p = X.shape
    
    init_beta = generate_initialization_of_beta( p, k, initial_sol_size )  # Generate initialization of beta
    
    results = [Discrete_First_Order( init_beta[i], k, X, y, 10 ** ( -3 ) ) for i in range( initial_sol_size )]
    
    optimal_sol, optimal_obj = None, np.inf
    for i in range( initial_sol_size ):
        discrete_sol, discrete_obj = results[i]
        
        if discrete_obj < optimal_obj:
            optimal_obj = discrete_obj
            optimal_sol = discrete_sol.copy()
    
    # w = 1 - z, z = 1 when beta != 0, w = 0 when beta != 0
    w_sol_indices = np.where( optimal_sol != 0 )[0]
    w_sol = np.ones( p )
    w_sol[w_sol_indices] = 0
    
    error_sol = y - np.dot( X, optimal_sol )
    
    initial_sol = {
        'beta': optimal_sol,
        'w': w_sol,
        'error': error_sol
    }
    
    # Use the solution from Discrete First-Order as the warm start of the problem 2.4
    obj_val, hybrid_sol = opt_value( X = X, y = y, k = k, initial_sol = initial_sol )
    
    return obj_val, hybrid_sol

def opt_value( X: np.array, y: np.array, k: int, initial_sol: dict = None, lower_bound: int = 5, OutputFlag: bool = False ) -> tuple[float, np.array]:
    """
    Solve the MIO problem 2.4.
    
    Parameters
    ----------
    X : np.array
        Feature matrix.
    y : np.array
        Target column.
    k : int
        The maximum number of feature selected.
    ***************************************************************
    initial_sol : dict, optional
        The initial solution for warm start. The default is None.
    lower_bound : int, optional
        The lower bound for decision variables. The default is 5.
    OutputFlag : bool, optional
        Whether to show the solving process of Gurobi or not. The default is False.
    
    Returns
    ----------
    obj : float
        Optimal objective function value.
    beta_sol : np.array
        Optimal solution of beta.
    
    Source:
    ----------
    Bertsimas, D., King, A., & Mazumder, R. (2016). Best subset selection via a modern optimization lens.
    Page 821, problem (2.4)
    """
    with gp.Env( empty = True ) as env:
        env.setParam( 'OutputFlag', 1 ) if OutputFlag else env.setParam( 'OutputFlag', 0 )
        env.start()
        with gp.Model( "opt_value", env = env ) as m:
            try:
                # Compute the lower bound for decision variables
                lb = y.min() - lower_bound * ( abs( y.min() ) + abs( y.max() ) )
                
                # Decision variables
                beta = m.addVars( X.shape[1], lb = lb, vtype = GRB.CONTINUOUS, name = "beta" )
                # change of variable w = 1 - z
                w = m.addVars( X.shape[1], vtype = GRB.BINARY, name = "w" )
                
                # Set warm start values
                if initial_sol is not None:
                    for i in range( X.shape[1] ):
                        beta[i].start = initial_sol['beta'][i]
                        w[i].start = initial_sol['w'][i]
                
                m.update()
                
                # Objective function
                obj = gp.QuadExpr()
                for i in range( X.shape[0] ):
                    obj += ( ( y[i] - gp.quicksum( X[i, j] * beta[j] for j in range( X.shape[1] ) ) ) \
                        * ( y[i] - gp.quicksum( X[i, j] * beta[j] for j in range( X.shape[1] ) ) ) )
                obj /= 2
                
                m.setObjective( obj, GRB.MINIMIZE )
                
                # Add SOS constraints of type 1: https://www.gurobi.com/documentation/8.1/examples/sos_py.html 
                # 1 - z[i] = 0 or beta[i] = 0
                # Please note that w = 1 - z
                for i in range( X.shape[1] ):
                    m.addSOS( GRB.SOS_TYPE1, [beta[i], w[i]], [1, 2] )
                
                # Summation of w constraint
                m.addConstr( w.sum() >= X.shape[1] - k )
                
                m.optimize()
                # m.write('Out/opt_value.lp')
                
                beta_sol  = []
                for v in m.getVars():
                    if "beta" in v.varName:
                        beta_sol.append( v.x )
                obj = m.objVal
                return obj, beta_sol
            except gp.GurobiError:
                print( 'Encountered a Gurobi error in opt_value' )

def ui_max( X: np.array, y: np.array, jth_feature: int, UB: float, warm_start: np.array = None, lower_bound: int = 5, OutputFlag: bool = False ) -> float:
    """
    Solve the convex optimization problem to receive the upper bound of the beta.
    
    Parameters
    ----------
    X : np.array
        Feature matrix.
    y : np.array
        Target column.
    jth_feature : int
        The index of the jth feature.
    UB : float
        The upper bound of the objective function.
    ***************************************************************
    warm_start : np.array, optional
        The warm start for beta. The default is None.
    lower_bound : int, optional
        The lower bound for decision variables. The default is 5.
    OutputFlag : bool, optional
        Whether to show the solving process of Gurobi or not. The default is False.
    
    Returns
    ----------
    obj : float
        Upper bound of the beta.
    
    Source:
    ----------
    Bertsimas, D., King, A., & Mazumder, R. (2016). Best subset selection via a modern optimization lens.
    Page 826, section 2.3.2, problem (2.14)
    """
    with gp.Env( empty = True ) as env:
        env.setParam( 'OutputFlag', 1 ) if OutputFlag else env.setParam( 'OutputFlag', 0 )
        env.start()
        with gp.Model( "ui_max", env = env ) as m:
            try:
                # Compute the lower bound for decision variable
                lb = y.min() - lower_bound * ( abs( y.min() ) + abs( y.max() ) )
                
                # Decision variables
                beta = m.addVars( X.shape[1], lb = lb, vtype = GRB.CONTINUOUS, name = "beta" )
                
                # Set warm start value receive from discrete first-order method
                if warm_start is not None:
                    for i in range( X.shape[1] ):
                        beta[i].start = warm_start['beta_start'][i]
                
                m.update()
                
                # Objective function
                m.setObjective( beta[jth_feature], GRB.MAXIMIZE )
                
                if UB is None:
                    # https://numpy.org/doc/stable/reference/generated/numpy.true_divide.html
                    # Return the quotient x1/x2
                    a = np.true_divide( y.max(), X[:,jth_feature] )
                    UB = np.max( np.abs( a[a != np.inf] ) ) 
                
                m.addConstr( ( gp.quicksum( ( y[i] - gp.quicksum( X[i, j] * beta[j] for j in range( X.shape[1] ) ) ) \
                                            * ( y[i] - gp.quicksum( X[i, j] * beta[j] for j in range( X.shape[1] ) ) ) for i in range( X.shape[0] ) ) / 2 ) <= 1.1 * UB )
                
                m.optimize()
                # m.write( 'Out/ui_max.lp' )
                if m.status == GRB.OPTIMAL:
                    return m.objVal
                else:
                    return None
            except gp.GurobiError:
                print('Encountered a Gurobi error in ui_max')
                return None

def ui_min( X: np.array, y: np.array, jth_feature: int, UB: float, warm_start: np.array = None, lower_bound: int = 5, OutputFlag: bool = False ) -> float:
    """
    Solve the convex optimization problem to receive the lower bound of the beta.
    
    Parameters
    ----------
    X : np.array
        Feature matrix.
    y : np.array
        Target column.
    jth_feature : int
        The index of the jth feature.
    UB : float
        The upper bound of the objective function.
    ***************************************************************
    warm_start : np.array, optional
        The warm start for beta. The default is None.
    lower_bound : int, optional
        The lower bound for decision variables. The default is 5.
    OutputFlag : bool, optional
        Whether to show the solving process of Gurobi or not. The default is False.
    
    Returns
    ----------
    obj : float
        Lower bound of the beta.
    
    Source:
    ----------
    Bertsimas, D., King, A., & Mazumder, R. (2016). Best subset selection via a modern optimization lens.
    Page 826, section 2.3.2, problem (2.14)
    """
    with gp.Env( empty = True ) as env:
        env.setParam( 'OutputFlag', 1 ) if OutputFlag else env.setParam( 'OutputFlag', 0 )
        env.start()
        with gp.Model( "ui_min", env = env ) as m:
            try:
                # Compute the lower bound for decision variable
                lb = y.min() - lower_bound * ( abs( y.min() ) + abs( y.max() ) )
                
                # Decision variables
                beta = m.addVars( X.shape[1], lb = lb, vtype = GRB.CONTINUOUS, name = "beta" )
                error = m.addVars( X.shape[0], lb = lb, vtype = GRB.CONTINUOUS, name = "error" )
                
                # Set warm start value receive from discrete first-order method
                if warm_start is not None:
                    for i in range( X.shape[1] ):
                        beta[i].start = warm_start['beta_start'][i]
                
                m.update()
                
                # Objective function
                m.setObjective( beta[jth_feature], GRB.MINIMIZE )      
                
                if UB is None:
                    # https://numpy.org/doc/stable/reference/generated/numpy.true_divide.html
                    # Return the quotient x1/x2
                    a = np.true_divide( y.max(), X[:,jth_feature] )
                    UB = np.max( np.abs( a[a != np.inf] ) ) 
                
                m.addConstr( ( gp.quicksum( ( y[i] - gp.quicksum( X[i, j] * beta[j] for j in range( X.shape[1] ) ) ) \
                                            * ( y[i] - gp.quicksum( X[i, j] * beta[j] for j in range( X.shape[1] ) ) ) for i in range( X.shape[0] ) ) / 2 ) <= 1.1 * UB )
                
                m.optimize()
                # m.write( 'Out/ui_min.lp' )
                if m.status == GRB.OPTIMAL:
                    return m.objVal
                else:
                    return None
            except gp.GurobiError:
                print('Encountered a Gurobi error in ui_min')
                return None

def specific_beta_bound( X: np.array, y: np.array, UB: float, j: int ) -> float:
    """
    Used to calculate the specific bound for the jth beta by using multiprocessing.
    
    Parameters
    ----------
    X : np.array
        Feature matrix.
    y : np.array
        Target column.
    UB : float
        The upper bound of the objective function.
    j : int
        The index of the jth feature.
    
    Returns
    ----------
    j_beta_bound : float
        The specific bound for the jth beta.
    """
    ui_positive = ui_max( X, y, j, UB )
    ui_negative = ui_min( X, y, j, UB )
    if ui_positive == None or ui_negative == None:
        j_beta_bound = np.nan
    else:
        j_beta_bound = max( abs( ui_positive ), abs( ui_negative ) )
    return j_beta_bound

'''
def beta_bound( X: np.array, y: np.array, k: int ) -> float:
    """
    Compute the upper bound of the beta. (Value of the big-M)
    
    Parameters
    ----------
    X : np.array
        Feature matrix.
    y : np.array
        Target column.
    k : int
        The number of features.
    
    Returns
    ----------
    beta_ub : float
        The upper bound of the beta.
    
    Source:
    ----------
    Bertsimas, D., King, A., & Mazumder, R. (2016). Best subset selection via a modern optimization lens.
    Page 824 Proposition 1,
    page 825, Section 2.3.1, Theorem 2.1 (b),
    page 826, Section 2.3.2, Bounds on beta_i's, problem (2.14),
    page 827, Section 2.3.3, bounds from advanced warm-starts.
    Based on the recommendations in page 827.
    """
    
    n, p = X.shape
    # Page 824, Proposition 1
    # mu := max_(i != j) | <X_i, X_j> |
    # Since the columns of X is normalized, compute the correlation coefficient matrix for the inner product.
    corr = pd.DataFrame( X ).corr().fillna( 0 ).to_numpy()
    
    # Replace the diagonal element with 0 since we want max_(i != j)
    np.fill_diagonal( corr, 0 )
    mu = np.max( np.abs( corr ) )
    gamma = 1 - mu * ( k - 1 )
    
    beta_ub_opt1 = -np.inf
    if gamma > 0: # We can use Theorem 2.1 (b) in page 825
        y_normalized = ( y - np.mean( y ) ) / np.linalg.norm( y - np.mean( y ) )
        # First compute the correlation between each X and y
        XY = np.column_stack( ( X, y_normalized ) )
        corr_matrix = pd.DataFrame( XY ).corr().fillna(0).to_numpy()
        XY_corr = corr_matrix[:-1, -1]
        XY_corr = XY_corr[np.abs( XY_corr ).argsort()[::-1]]
        
        # Compute the upper bound for beta
        beta_ub_opt1 = np.min( [( np.sqrt( sum( XY_corr[i]**2 for i in range( k ) ) ) / gamma ), ( np.linalg.norm( y ) / np.sqrt( gamma ) )] )
    
    # Use specific bound in Section 2.3.2 in page 826, problem 2.14.
    beta_ub_opt2 = -np.inf
    UB, hybrid_sol = Warm_Start_MIO( X = X, y = y, k = k )
    if n > p: # problem 2.14 is not suitable for p > n regime
        # Need to run ui_max and ui_min for each feature
        cpus = round( cpu_count() / 2 )
        with Pool( processes = cpus ) as pool:
            sol_pools = pool.starmap( specific_beta_bound, [( X, y, UB, j ) for j in range( X.shape[1] )] )
        beta_ub_opt2 = max( sol_pools ) # Choose the maximum one as the upper bound for beta
    
    # Use specific bound in Section 2.3.3 in page 827
    beta_ub_opt3 = 2 * np.linalg.norm( hybrid_sol, np.inf )
    
    # Choose the tighter bound for the final beta bound
    beta_ub = np.max( [beta_ub_opt1, beta_ub_opt2, beta_ub_opt3] )
    return beta_ub
'''

def max_k( X: np.array, rho: float = 0.8, OutputFlag: bool = False ) -> int:
    """
    Solve the MIO problem to receive the upper bound of k.
    
    Parameters
    ----------
    X : np.array
        Feature matrix.
    ***************************************************************
    rho : float, optional
        The threshold for the highly correlated between 2 regressors. The default is 0.8.
    OutputFlag : bool, optional
        Whether to show the solving process of Gurobi or not. The default is False.
    
    Returns
    ----------
    k_max : int
        The upper bound of k.
    
    Source:
    ----------
    Bertsimas, D., & King, A. (2016). OR forum—an algorithmic approach to linear regression. Operations Research, 64(1), 2-16.
    Page 7. Section 4.1. Preprocessing.
    """
    with gp.Env( empty = True ) as env:
        env.setParam( 'OutputFlag', 0 )
        env.start()
        with gp.Model( "max_k", env = env ) as m:
            try:
                # Decision variables
                z = m.addVars( X.shape[1], vtype = GRB.BINARY, name = "z" )
                
                m.update()
                
                # Objective function
                obj = z.sum()
                m.setObjective( obj, GRB.MAXIMIZE )
                # if the correlation between 2 regressors is large, then add one variable at most
                corr_matrix = pd.DataFrame( X ).corr().fillna( 0 ).to_numpy() # Use pandas .corr() and fill NAN with 0.
                for i in range( corr_matrix.shape[0] - 1 ):
                    for j in range( i + 1, corr_matrix.shape[0] ):
                        if abs( corr_matrix[i, j] ) >= rho:
                            m.addConstr( z[i] + z[j] <= 1 )
                
                m.optimize()
                k_max = math.ceil( m.objVal )
                return k_max
            
            except gp.GurobiError:
                print('Encountered a Gurobi error in max_k')

def Holistic_Regression( X: np.array, y: np.array, k: int, Gamma: float, M: float = 500000000\
                        , time_limit: int = 20, z_pos: list = [], pairwise_HC: list = [], group_sparsity: list = [], OutputFlag: bool = False ):
    """
    Fit Holistic regression model.
    
    Parameters
    ----------
    X : np.array
        Feature matrix.
    y : np.array
        Target vector.
    k : int
        The upper bound of k.
    Gamma : float
        The threshold for the sum of absolute values of the regression coefficients.
    ***************************************************************
    M : float, optional
        The upper bound for the regression coefficients. The default is None.
    time_limit : int, optional
        The time limit for the model. The default is 20.
    rho : float, optional
        The threshold for the highly correlated between 2 regressors. The default is 0.8.
    z_pos : list, optional
        The list of the detected multicollinearity relationships. The default is [].
    OutputFlag : bool, optional
        Whether to show the solving process of Gurobi or not. The default is False.
    
    Returns
    ----------
    beta : np.array
        The optimal solution of the beta.
    obj_val: float
        The optimal objective function value.
    execute_time: float
        The execution time of solving the model.
    
    Source:
    ----------
    Bertsimas, D., & Li, M. L. (2020). Scalable holistic linear regression. Operations Research Letters, 48(3), 203-208.
    Page 204, problem (1).
    """
    with gp.Env( empty = True ) as env:
        env.setParam( 'OutputFlag', 0 )
        env.start()
        with gp.Model( "Holistic_regression", env = env ) as m:
            # Initialize return variables
            beta_sol, obj_val, execute_time, time_limit_reached = None, None, None, False
            
            # Setup model parameters
            m.Params.OutputFlag = 1 if OutputFlag else 0
            m.Params.TimeLimit = time_limit
            m.Params.LazyConstraints = 1
            m.Params.IntFeasTol = 10**(-9)
            
            # Decision variables
            beta = m.addVars( X.shape[1], lb = -M, ub = M, vtype = GRB.CONTINUOUS, name = "beta" )
            beta_abs = m.addVars( X.shape[1], lb = 0.0, vtype = GRB.CONTINUOUS, name = "beta_abs" )
            z = m.addVars( X.shape[1], vtype = GRB.BINARY, name = "z" )
            w = m.addVars( X.shape[1], vtype = GRB.BINARY, name = "w" )
            
            # Objective function
            # obj = (1/2) * ||y - X*beta||_2^2 + Gamma * ||beta||_1
            obj = gp.QuadExpr()
            for i in range( X.shape[0] ):
                obj += ( ( y[i] - gp.quicksum( X[i, j] * beta[j] for j in range( X.shape[1] ) ) ) * \
                        ( y[i] - gp.quicksum( X[i, j] * beta[j] for j in range( X.shape[1] ) ) ) )
            obj /= 2
            obj += ( Gamma * beta_abs.sum() )
            m.setObjective( obj, GRB.MINIMIZE )
            
            # beta_abs = | beta | constraints
            for j in range( X.shape[1] ):
                m.addGenConstrAbs( beta_abs[j], beta[j] )
            
            # SOS-1 constraints between z and beta
            for j in range( X.shape[1] ):
                m.addConstr( w[j] == 1 - z[j] )
                m.addSOS( GRB.SOS_TYPE1, [w[j], beta[j]], [1, 2] )
            
            # General sparsity constraint.
            m.addConstr( z.sum() <= k + 1 )
            
            # Interception term must be selected.
            m.addConstr( z[0] == 1 )
            
            # Adding pairwise highly correlated constraints.
            if len( pairwise_HC ) != 0:
                for hc_idx in range( len( pairwise_HC ) ):
                    m.addConstr( gp.quicksum( z[_] for _ in pairwise_HC[hc_idx] ) <= 1 )
            
            # Adding group sparsity constraints.
            if len( group_sparsity ) != 0:
                for sparsity_idx in range( len( group_sparsity ) ):
                    for idx_ in range( len( group_sparsity[sparsity_idx] ) - 1 ):
                        m.addConstr( z[ group_sparsity[sparsity_idx][idx_] ] == z[ group_sparsity[sparsity_idx][idx_ + 1] ] )
            
            # Define callback function for adding multicollinearity constraints.
            def my_Callback( model, where ):
                if where == GRB.Callback.MIPSOL:
                    # Obtain the current solution of beta
                    beta_sol = model.cbGetSolution( beta )
                    non_zero_beta = []
                    for key, value in beta_sol.items():
                        if round( value, 5 ) != 0:
                            non_zero_beta.append( key )
                    
                    # Add multicollinearity constraints if needed.
                    if len( z_pos ) != 0:
                        for z_pos_idx in range( len( z_pos ) ):
                            # Check if the current solution contains all the features in z_pos[z_pos_idx]
                            intersection_ = set( non_zero_beta ).intersection( set( z_pos[z_pos_idx] ) )
                            if sorted( list( intersection_ ) ) == sorted( z_pos[z_pos_idx] ):
                                model.cbLazy( gp.quicksum( z[_] for _ in z_pos[z_pos_idx] ) <= len( z_pos[z_pos_idx] ) - 1 )
            
            start = time.time()
            m.optimize( my_Callback )
            end = time.time()
            execute_time = end - start
            
            if m.status == GRB.OPTIMAL:
                obj_val = m.objVal
                beta_sol = []
                for v in m.getVars():
                    if 'beta' in v.varName and 'beta_abs' not in v.varName:
                        beta_sol.append( np.round( v.x, 6 ) )
                        # beta_sol.append( v.x )
                
            elif m.status == GRB.TIME_LIMIT:
                time_limit_reached = True
            else:
                print( f'Gurobi status: {m.status}' )
            
            return beta_sol, obj_val, execute_time, time_limit_reached

def Performance_Index( X: np.array, y: np.array, beta: np.array, y_pred: np.array = None ) -> tuple[float, float, float]:
    """
    Compute the performance index for the given beta.
    
    Parameters
    ----------
    X : np.array
        Feature matrix.
    y : np.array
        Target column.
    beta : np.array
        The solution of beta.
    
    Returns
    ----------
    MSE : float
        Mean squared error.
    MAE : float
        Mean absolute error.
    R2_score : float
        R2 score.
    """
    #TODO: Check if this function works for multivariate Y
    if len( y.shape ) != 1: # Multivariate Y
        new_beta = np.zeros( ( X.shape[1], y.shape[1] ) )
        for i in range( y.shape[1] ):
            new_beta[:,i] = beta
        beta = new_beta
    if y_pred is None:
        y_pred = np.dot( X, beta )
    MSE = mean_squared_error( y, y_pred )
    MAE = mean_absolute_error( y, y_pred )
    R2_score = r2_score( y, y_pred )
    return MSE, MAE, R2_score

def Simulation_Data( n: int, p: int, k_true: int, rho: float, noise_scale: float, SNR: float, MR: list, rand_seed: int, intercept: bool = True ):
    """
    Generate experimental data.
    
    Parameters
    ----------
    n : int
        Number of samples.
    p : int
        Number of features.
    k_true : int
        Number of the nonzero elements in true beta.
    rho : float
        Correlation coefficient between features.
    noise_scale : float
        Standard deviation of the Gaussian noise added to the multicollinear relationships.
    SNR : float
        Signal-to-noise ratio.
    MR : list
        Number of multicollinearity relationships.
    rand_seed : int
        Random seed.
    intercept : bool, optional
        Whether to consider intercept term in the target. The default is True.
    
    Returns
    ----------
    X : np.array
        Feature matrix.
    y : np.array
        Target column.
    true_beta_rec : list
        The true regression coefficients.
    indices_rec : list
        The indices of the multicollinearity relationships.
    feature_col : np.array
        The feature column names.
    """
    random.seed( int( rand_seed ) )
    rng = np.random.default_rng( seed = rand_seed )
    
    #! Generate X with pairwise correlation coefficient rho
    X = rng.normal( loc = 0.0, scale = 1.0, size = ( n, p + 1 ) )
    for i in range( 1, p ):
        X[:, i] = X[:, i] + rho * X[:, i - 1]
    
    #! Generate multicollinear relationships
    MR2, MR3, MR4, MR8, MR10 = MR
    no_var = 2 * MR2 + 3 * MR3 + 4 * MR4 + 8 * MR8 + 10 * MR10
    
    # Multicollinear relationship coefficients
    gamma = rng.uniform( -10, 10, size = no_var ).tolist()
    
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
    
    # Add noise to feature matrix
    noise = rng.normal( loc = 0.0, scale = noise_scale, size = ( n, p + 1 ) )
    X = X + noise
    
    #! Generate true beta
    true_beta = np.zeros( p )
    for i in range( p ):
        if i % ( p // k_true ) == 0:
            true_beta[i] = 1
    
    #! Add interception term into feature matrix
    X[:, 0] = 1
    feature_col = np.array( ['Intercept'] + [f'X{i + 1}' for i in range( p )] )
    if intercept:
        true_beta = np.append( 1, true_beta )
    else:
        true_beta = np.append( 0, true_beta )
    
    #! Generate Y based on the SNR value
    signal = np.dot( X, true_beta )
    noise = rng.normal( size = n )
    noise *= ( np.linalg.norm( signal ) / np.linalg.norm( noise ) ) / SNR  # Scaling according to SNR
    Y = signal + noise
    
    #! Transfer X and Y into pandas DataFrame
    X = pd.DataFrame( X, columns = feature_col )
    Y = pd.DataFrame( Y, columns = ['Y'] )
    
    return X, Y, true_beta, indices, feature_col

def KF_Score( true_beta: np.array, beta: np.array ):
    ACC, FPR = 0, 0
    true_beta = true_beta[1:]
    beta = beta[1:]
    true_indices = (np.where( true_beta != 0 )[0]).tolist()
    beta_indices = (np.where( beta != 0 )[0]).tolist()
    print( len( set( beta_indices ) ) )
    # Accuracy
    ACC = ( ( len( set( true_indices ) & set( beta_indices ) ) ) / ( len( set( true_indices ) ) ) ) * 100
    # False positive rate
    FPR = ( ( len( set( beta_indices ) - set( true_indices ) ) ) / ( len( set( beta_indices ) ) ) ) * 100
    return ACC, FPR