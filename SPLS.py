import os, time
if os.path.exists( 'C:/Program Files/R/R-4.4.2' ):
    os.environ['R_HOME'] = 'C:/Program Files/R/R-4.4.2'
elif os.path.exists( 'C:/Program Files/R/R-4.4.3' ):
    os.environ['R_HOME'] = 'C:/Program Files/R/R-4.4.3'
elif os.path.exists( 'C:/Program Files/R/R-4.4.1' ):
    os.environ['R_HOME'] = 'C:/Program Files/R/R-4.4.1'

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

import pandas as pd, numpy as np

# Import the R package
spls = importr('spls')

def SPLS( X_train, y_train, X_test, y_test, K, eta ):
    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
        X_train = ro.conversion.py2rpy( pd.DataFrame( X_train ) )
        y_train = ro.conversion.py2rpy( pd.DataFrame( y_train ) )
        X_test = ro.conversion.py2rpy( pd.DataFrame( X_test ) )
        y_test = ro.conversion.py2rpy( pd.DataFrame( y_test ) )
    
    ro.globalenv['X_train'] = ro.FloatVector( X_train )
    ro.globalenv['y_train'] = ro.FloatVector( y_train )
    ro.globalenv['X_test'] = ro.FloatVector( X_test )
    ro.globalenv['y_test'] = ro.FloatVector( y_test )
    ro.globalenv['K'] = int( K )
    ro.globalenv['eta'] = float( eta )
    
    start = time.time()
    ro.r('''
        library(spls)
        
        # Fit the model with optimal K and eta
        model <- spls(X_train, y_train, K = K, eta = eta, select = "pls2", fit = "simpls", scale.x = FALSE, scale.y = FALSE)
        beta <- coef(model)
        train_pred <- matrix(predict(model, X_train), ncol = length(y_train))
        test_pred <- matrix(predict(model, X_test), ncol = length(y_test))
    ''')
    
    end = time.time()
    time_taken = end - start
    train_pred = np.array( ro.globalenv['train_pred'] ).reshape( -1, ).tolist()
    test_pred = np.array( ro.globalenv['test_pred'] ).reshape( -1, ).tolist()
    beta = np.array( ro.globalenv['beta'] ).reshape( -1, ).tolist()
    return train_pred, test_pred, beta, time_taken

def SPLS_CV( X, y ):
    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
        X = ro.conversion.py2rpy( pd.DataFrame( X ) )
        y = ro.conversion.py2rpy( pd.DataFrame( y ) )
    
    ro.globalenv['X'] = ro.FloatVector( X )
    ro.globalenv['y'] = ro.FloatVector( y )
    
    ro.r('''
        library(spls)
        
        X <- X
        y <- y
        
        # Grid search for optimal K and eta
        best_mse <- Inf
        for (eta in seq(0.1, 0.9, 0.1)){
            if (dim(X)[1] > dim(X)[2]){
                K_ub <- ncol(X)
            }else{
                K_ub <- nrow(X)-1
            }
            for (K in seq(1, K_ub, 1)){
                model <- spls(X, y, K = K, eta = eta, select = "simpls", fit = "simpls", scale.x = FALSE, scale.y = FALSE)
                pred <- list(predict(model, X))
                mse <- mean(unlist((y - pred)^2))
                if (mse < best_mse) {
                    best_mse <- mse
                    best_K <- K
                    best_eta <- eta
                }
            }
        }
        # Fit the model with optimal K and eta
        model <- spls(X, y, K = best_K, eta = best_eta, select = "pls2", fit = "simpls", scale.x = FALSE, scale.y = FALSE)
        # Compute MSE
        pred <- matrix(predict(model, X), ncol = length(y))
        beta <- coef(model)
    ''')
    y_pred = np.array( ro.globalenv['pred'] ).reshape( -1, )
    beta = np.array( ro.globalenv['beta'] ).reshape( -1, )
    optimal_K = ro.globalenv['best_K'][0]
    optimal_eta = ro.globalenv['best_eta'][0]
    return y_pred, beta, optimal_K, optimal_eta