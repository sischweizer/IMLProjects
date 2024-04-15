# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, Exponentiation, ExpSineSquared, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import GaussianNB

def plot_data(label,data):
    
    time = np.arange(len(label))
    #data = np.power(10, data)
    plt.plot(time, data)
    plt.xlabel('season')
    plt.ylabel('price')
    plt.show()

def fill_data(data):
    #imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = KNNImputer(n_neighbors=2, weights="uniform")
    data = imp.fit_transform(data)
    return data

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test

    # dataframe to numpy  
    #drop(['price_CHF'],axis=1)
    X_train = train_df.drop(['season'],axis=1)
    X_test = test_df.drop(['season'],axis=1)
    
    # label for plot
    label_train = train_df['season']
    label_test = test_df['season']
    
    ### data insertion ###
    X_train = fill_data(X_train)
    X_test = fill_data(X_test)


    X_train = pd.DataFrame(X_train, columns=train_df.drop(['season'],axis=1).columns)
    y_train = X_train['price_CHF']
    X_train = X_train.drop(['price_CHF'],axis=1)
    
    #visualization 
    plot = False
    if(plot == True):
        plot_data(label_train, X_train)
        plot_data(label_train, y_train)
        plot_data(label_test, X_test)
    
    # Dummy initialization of the X_train, X_test and y_train
    # TODO: Depending on how you deal with the non-numeric data, you may want to 
    # modify/ignore the initialization of these variables   
    #X_train = np.zeros_like(train_df.drop(['price_CHF'],axis=1))
    #y_train = np.zeros_like(train_df['price_CHF'])
    #X_test = np.zeros_like(test_df)
    

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    y_pred=np.zeros(X_test.shape[0])
    #TODO: Define the model and fit it using training data. Then, use test data to make predictions

    
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
    from sklearn.metrics import r2_score
    
    kernels = [Matern(), RationalQuadratic()]
    
    
    #crossvalidation
    
    kf = KFold(n_splits=5)

    
    
    #Mattern Crossvalidation 
    # quadreatic crossvalidation
    # Crossvalidation Selection 
    
    
    lengths = [0.001,0.1,1,10,100]
    matern_error_mat = np.zeros((5, len(lengths)))
    print(X_train.shape)
    print("Matern CV")
    for (j, length) in enumerate(lengths):
        

        for i, (train, test) in enumerate(kf.split(X_train)): 
            print("CV set ", i)
            X_train_CV = X_train.iloc[train]
            y_train_CV = y_train.iloc[train]
            
            X_test_CV = X_train.iloc[test]
            y_test_CV = y_train.iloc[test]

            model = GaussianProcessRegressor(Matern(length_scale=length))
            model.fit(X_train_CV, y_train_CV)
            matern_error_mat[i][j] = r2_score(y_test_CV, model.predict(X_test_CV))
            
    print(matern_error_mat)
        
    avg_Matern_error = np.mean(matern_error_mat, axis=0)
    print(avg_Matern_error)

    length_pos = np.where(avg_Matern_error == avg_Matern_error.max())[0][0]
    print(length_pos)
    
    
    
    
    print("RationalQuadratic CV")
    alphas = [0.001,0.1,1,10,100]
    RQ_error_mat = np.zeros((5, len(alphas)))    
    
    for (j, alpha) in enumerate(alphas):
        

        for i, (train, test) in enumerate(kf.split(X_train)): 
            print("CV set ", i)
            X_train_CV = X_train.iloc[train]
            y_train_CV = y_train.iloc[train]
            
            X_test_CV = X_train.iloc[test]
            y_test_CV = y_train.iloc[test]

            model = GaussianProcessRegressor(RationalQuadratic(alpha=alpha))
            model.fit(X_train_CV, y_train_CV)
            RQ_error_mat[i][j] = r2_score(y_test_CV, model.predict(X_test_CV))
            

    print(RQ_error_mat)
    avg_RQ_error = np.mean(RQ_error_mat, axis=0)
    print(avg_RQ_error)

    alpha_pos = np.where(avg_RQ_error == avg_RQ_error.max())[0][0]
    print(alpha_pos)

    if (avg_RQ_error[alpha_pos] < avg_Matern_error[length_pos]):
        model = GaussianProcessRegressor(Matern(length_scale=lengths[length_pos]))
        print("Matern chosen")
    else:
        model = GaussianProcessRegressor(RationalQuadratic(alpha=alphas[alpha_pos]))
        print("Rational Quadratic chosen")

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    print(y_pred)


    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

