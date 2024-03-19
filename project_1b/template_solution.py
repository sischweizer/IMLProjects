# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# Add any additional imports here (however, the task is solvable without using 
# any additional imports)
# import ...

def transform_data(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component of X) 
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant feature: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: array of floats: dim = (700,21), transformed input with 21 features
    """
    X_transformed = np.zeros((700, 21))
    # TODO: Enter your code here

    #print(X[0])
    #print(y[0])
    
    #X_n, y_n = normalise(X,y)

    #print("normalise data: ")
    #print(X_n[0])
    #print(y_n[0])

    for i, row in enumerate(X):
        x_i = np.zeros((21,))
        x_i[0:5] = row
        x_i[5:10] = np.square(row)
        x_i[10:15] = np.exp(row)
        x_i[15:20] = np.cos(row)
        x_i[20] = 1

        X_transformed[i] = x_i 
        
    #print(X_transformed[0])

    assert X_transformed.shape == (700, 21)
    return X_transformed


def fit(X, y):
    """
    This function receives training data points, transforms them, and then fits the linear regression on this 
    transformed data. Finally, it outputs the weights of the fitted linear regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of floats, dim = (700,), input labels)

    Returns
    ----------
    w: array of floats: dim = (21,), optimal parameters of linear regression
    """
    w = np.zeros((21,))
    X_transformed = transform_data(X)
    # TODO: Enter your code here

    #crossvalidation
    lambdas = [0.1, 0.3, 0.5, 0.7, 1, 2, 5, 10, 20]
    kf = KFold(n_splits=5)

    error_mat = np.zeros((5, len(lambdas)))

    for (j, lam) in enumerate(lambdas):
        print("lambda: ", lam)

        for i, (train, test) in enumerate(kf.split(X)): 
            print("CV set ", i)
            X_train = X_transformed[train]
            y_train = y[train]
        
            X_test = X_transformed[test]
            y_test = y[test]


            model = Ridge(alpha=lam, fit_intercept=False).fit(X_train,y_train)
            w = model.coef_
            error_mat[i][j] = np.sqrt(np.mean(np.square(np.matmul(X_test, w) - y_test)))

    avg_error = np.mean(error_mat, axis=0)
    print(avg_error)

    lam_final = lambdas[np.where(avg_error == avg_error.min())[0][0]]
    print("lambda final: ", lam_final)
    
    model = Ridge(alpha=lam_final, fit_intercept=False).fit(X_transformed,y)
    w = model.coef_
    training_error = np.sqrt(np.mean(np.square(np.matmul(X_transformed, w) - y)))
    print(training_error)

    '''
    best_validation_loss = 1000000
    for i, (train, test) in enumerate(kf.split(X_transformed)): 
            print("CV set ", i)
            X_train = X_transformed[train]
            y_train = y[train]
            
            X_test = X_transformed[test]
            y_test = y[test]
        
            model = LinearRegression(fit_intercept=False).fit(X_train,y_train)
            
            validation_loss = np.sqrt(np.mean(np.square(model.predict(X_test) - y_test)))
            
            if (validation_loss < best_validation_loss):
                best_validation_loss = validation_loss
                w = model.coef_
                
                print("best validation loss: ", validation_loss)
    '''

    
    
    #original code
    #model = Ridge(alpha=0.5, fit_intercept=False).fit(X_transformed,y)
    #w = model.coef_
    #training_error = np.sqrt(np.mean(np.square(np.matmul(X_transformed, w) - y)))
    #print(training_error)

    #comment
    #model = LinearRegression(fit_intercept=False).fit(X_transformed,y)

    assert w.shape == (21,)
    return w


#def normalise(X, y):
    #std = StandardScaler()
    #data = np.concatenate((X, y.reshape((len(y), 1))), axis=1)
    #std.fit(data)
    #return (std.transform(X), std.transform(y))


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("project_1b/train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    w = fit(X, y)
    # Save results in the required format
    np.savetxt("project_1b/results.csv", w, fmt="%.12f")
