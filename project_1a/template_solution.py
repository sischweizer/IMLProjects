# This serves as a template which will guide you through the implementation of this task. It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# First, we import necessary libraries:
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge

# Add any additional imports here (however, the task is solvable without using 
# any additional imports)
# import ...

def fit(X, y, lam):
    """
    This function receives training data points, then fits the ridge regression on this data
    with regularization hyperparameter lambda. The weights w of the fitted ridge regression
    are returned. 

    Parameters
    ----------
    X: matrix of floats, dim = (135,13), inputs with 13 features
    y: array of floats, dim = (135,), input labels)
    lam: float. lambda parameter, used in regularization term

    Returns
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression
    """
    print("fit")
    w = np.zeros((13,))
    # TODO: Enter your code here
    print(lam)
    rigde = Ridge(alpha=lam)
    
    rigde.fit(X,y)

    w = rigde.coef_

    
    #sklearn.linear_model.Ridge(alpha=lam)

    assert w.shape == (13,)
    return w


def calculate_RMSE(w, X, y):
    """This function takes test data points (X and y), and computes the empirical RMSE of 
    predicting y from X using a linear model with weights w. 

    Parameters
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression 
    X: matrix of floats, dim = (15,13), inputs with 13 features
    y: array of floats, dim = (15,), input labels

    Returns
    ----------
    RMSE: float: dim = 1, RMSE value
    """
    RMSE = 0
    # TODO: Enter your code here
    
    print(w)
    pred = np.matmul(X, w)
    print(pred)
    error = pred - y 
    print(error)
    sq_error = np.square(error)
    print(sq_error)
    RMSE = np.mean(sq_error)
    
    assert np.isscalar(RMSE)
    return RMSE


def average_LR_RMSE(X, y, lambdas, n_folds):
    """
    Main cross-validation loop, implementing 10-fold CV. In every iteration (for every train-test split), the RMSE for every lambda is calculated, 
    and then averaged over iterations.
    
    Parameters
    ---------- 
    X: matrix of floats, dim = (150, 13), inputs with 13 features
    y: array of floats, dim = (150, ), input labels
    lambdas: list of floats, len = 5, values of lambda for which ridge regression is fitted and RMSE estimated
    n_folds: int, number of folds (pieces in which we split the dataset), parameter K in KFold CV
    
    Returns
    ----------
    avg_RMSE: array of floats: dim = (5,), average RMSE value for every lambda
    """
    RMSE_mat = np.zeros((n_folds, len(lambdas)))

    # TODO: Enter your code here. Hint: Use functions 'fit' and 'calculate_RMSE' with training and test data
    # and fill all entries in the matrix 'RMSE_mat'
    w = []
    kf = KFold(n_splits=n_folds)
    kf.get_n_splits(X)
    for i, (train_index, test_index) in enumerate(kf.split(X)): 
        
        
        X_split = np.array([X[k] for k in train_index]).reshape(135,13)
        y_split = np.array([y[k] for k in train_index])
        
        ValX_split = np.array([X[k] for k in test_index]).reshape(15,13)
        Valy_split = np.array([y[k] for k in test_index])
        
        #print(X)
        #print(X_split[0])
        #print(ValX_split[0])
        #print(y)
        #print(y_split)
        #print(Valy_split)
        
        for j in range(0,len(lambdas)):
            #w.append(fit(X_split, Y_split, lambdas[i]))
            coef = fit(X_split, y_split, lambdas[j])
            RMSE_mat[i:j] = calculate_RMSE(coef, ValX_split, Valy_split)
            
        
            
    
        #ridge = Ridge(alpha=lambdas[i])
        #result = cross_val_score(ridge,X,y,scoring="neg_mean_squared_error",cv=n_folds)
        #print(result)
        #print(result.transpose())
        #print(RMSE_mat)
        #RMSE_mat[i] = result
        

    

    avg_RMSE = np.mean(RMSE_mat, axis=0)
    assert avg_RMSE.shape == (5,)
    return avg_RMSE


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    print("loading data")
    data = pd.read_csv("project_1a/train.csv")
    print("done")
    y = data["y"].to_numpy()
    data = data.drop(columns="y")
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function calculating the average RMSE
    lambdas = [0.1, 1, 10, 100, 200]
    n_folds = 10
    avg_RMSE = average_LR_RMSE(X, y, lambdas, n_folds)
    # Save results in the required format
    np.savetxt("./results.csv", avg_RMSE, fmt="%.12f")
