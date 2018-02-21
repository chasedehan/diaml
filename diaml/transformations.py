#transformations.py
from sklearn.base import TransformerMixin, clone
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.pipeline import make_pipeline
#The objective here is to transform a certain subset of variables for a more linear representation.


# # Make log transformations
# from sklearn.preprocessing import FunctionTransformer
#     # we use this because it has a transform method already associated with it
#
class DiaLog(TransformerMixin):
    def __init__(self, cutoff=None):
        self.cutoff = cutoff
        print("Class is being built.  This does Nothing Right Now")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


# Polynomial Features

class DiaPoly(TransformerMixin):
    """
    Class does polynomial interpolation to straighten an arbitrary set of points
    """
    def __init__(self, n_folds=3, replace=True, degrees=None, reg=None, subset_col=False):
        self.reg = reg
        self.n_folds = n_folds
        self.replace = replace
        self.subset_col = subset_col

        if reg is not None:
            if type(reg) is not bool:
                raise ValueError("'reg' must be True, False, or left to its defaults")
        if type(replace) is not bool:
            raise ValueError("'replace' must be True or False")
        if type(subset_col) is not bool:
            raise ValueError("'subset_col' must be True or False")
        if n_folds < 2:
            raise ValueError('n_folds must be > 1')
        if degrees is None:
            self.degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            # self.degrees = [1, 2, 3]

    def _subset_columns(self, X):
        if self.subset_col:
            columns = X.columns
            run_columns = []
            for col in columns:
                this_col = X[col]
                if this_col.dtype in ['int32', 'int64', 'float32', 'float64']:
                    if len(this_col.value_counts()) > 2:
                        run_columns.append(col)
            self.run_columns = run_columns
        else:
            self.run_columns = X.columns

    def fit(self, X, y):

        #Check if should be classification or regression if user didn't pass any values
        if self.reg is None:
            if len(np.unique(y)) == 1:
                raise ValueError("y only has one value, cannot fit model")
            elif len(np.unique(y)) == 2:
                self.reg = False
            else:
                self.reg = True

        self._subset_columns(X)
        kf = KFold(n_splits=self.n_folds)
        self.col_degrees = {}
        self.models = {}
        for col_name in self.run_columns:
            this_col = X[[col_name]]
            scores = {}
            for degree in self.degrees:
                cv_scores = []
                for train_index, test_index in kf.split(this_col):
                    X_train, X_test = this_col.iloc[train_index], this_col.iloc[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    # Next is to train and evaluate the model performance on regression or classification
                    if self.reg:
                        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
                    else:
                        model = make_pipeline(PolynomialFeatures(degree), LogisticRegression())
                    model.fit(X_train, y_train)
                    y_hat = model.predict(X_test)
                    if self.reg:
                        cv_scores.append(mean_squared_error(y_test, y_hat))
                    else:
                        cv_scores.append(log_loss(y_test, y_hat))

                scores[degree] = np.mean(cv_scores)
            best_degree = min(scores.keys(), key=(lambda k: scores[k]))
            self.col_degrees[col_name] = best_degree
            model = make_pipeline(PolynomialFeatures(best_degree), LinearRegression())
            model.fit(this_col, y)
            self.models[col_name] = model
        return self

    def transform(self, X):
        if self.models is None:
            raise ValueError("You need to fit the model first")

        this_X = X.copy()

        transform_cols = self.col_degrees
        if self.replace:
            for col_name in transform_cols:
                this_col = this_X[[col_name]].copy()
                yhat = self.models[col_name].predict(this_col)
                this_X[col_name] = yhat
        else:
            for col_name in transform_cols:
                this_col = this_X[[col_name]].copy()
                yhat = self.models[col_name].predict(this_col)
                this_X[str(col_name)+'_pi'] = yhat           #insert a new column into the dataframe
        return this_X



#####Checking distributions

# Skewness - determining the variables to apply Box Cox to
# identify the variables to transform
# from scipy.stats import skew


class DiaSkew(TransformerMixin):
    def __init__(self, cutoff=None):
        self.cutoff = cutoff
        print("Class is being built.  This does Nothing Right Now")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


# skewness = skew(data)
# #Split out those above threshold
# transform = skewness > threshold
# #applying box cox to certain variables
# from scipy.stats import boxcox
# boxcox_data = boxcox(transform)
# #Put back together
# cbind(boxcox_data, non_boxcox)
#
# #Center-Scale - this is necessary in many regularized models
#     # Demean and look like normally distributed data
# from sklearn import preprocessing
# preprocessing.scale(X_train) #Does this have a fit/transform?
#
# #Can scale features according to a range, i.e. 0-1
#     #Why? preserves zero entries in sparse data
# from sklearn import preprocessing
# min_max_scaler = preprocessing.MinMaxScaler()
# X_train_minmax = min_max_scaler.fit_transform(X_train)


#PCA transformations
#It is not enough to center scale independently, can use PCA to remove linear dependence





#Another get dummies function - can probably handle it better above
def GetCatDummies(X, ReplaceNA=True):
    """Takes in a pd.DataFrame, checks for the categorical variables and returns a OHE DataFrame
    This needs to be done after any ordinal transformations because the original variable is replaced"""
    categoricals = []
    for col, col_type in X.dtypes.iteritems():
         if col_type == 'O':
              categoricals.append(col)
         else:
              X[col].fillna(0, inplace=True)
    return pd.get_dummies(X, columns=categoricals, dummy_na=ReplaceNA) #Default set to True


##### Transformations on Y
#Target Variable transformation
    #How do I automate this transformation?
    #One thought is to create a bunch of different transformations on Y, then run a simple lasso, comparing the R2
        #Then, keep the best representation
        #Only want to do this on regression tasks

    #Also have scale() and StandardScalar() to scale the target variable
#log, box-cox, etc?
    #Although, should that be done on the front end before the other transformations


