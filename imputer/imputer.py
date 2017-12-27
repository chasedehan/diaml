import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

########################################################################################
#
# Main Class and Methods
#
########################################################################################

class ImputeMissing(BaseEstimator, TransformerMixin):

    def __init__(self, cutoff=0.05, cont_impute="median", ohe=True, silent=False, ignore_vars=[]):
        self.cutoff = cutoff
        self.cont_impute = cont_impute
        self.ohe = ohe
        self.silent = silent
        self.ignore_vars = ignore_vars

        # Throw errors if the inputted parameters don't meet the necessary criteria
        if (cutoff <= 0) | (cutoff >=1):
            raise ValueError('cutoff should be between 0 and 1. You entered' + str(cutoff))

        # Issue warnings for parameters to still let it run
        if(cont_impute not in ["mean", "median", "most_frequent"]):
            #Changes to something that will run, but does raise an error to user
            self.cont_impute = "median"
            warnings.warn("WARNING: " + str(cont_impute) + " is not a valid value. Reverting back to 'median'")

    def fit(self, X, y=None):
        #Does not require a y, but will accept it to conform to sklearn methods
        #Check if there are any NA values
        if X.isnull().values.any():
            self.new_df = _imputer(X)  #TODO: this isn't quite right, what do we want to return?
        else:
            self.new_df = X
            warnings.warn("No NA values in dataframe")
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


########################################################################################
#
# Helper Functions to do the Heavy Lifting
#
########################################################################################

def _imputer(X):

    #What columns have missing values?
    missing_columns = []


    ########Create the new columns for those with missing values
    #if above threshold, create new columns
    new_columns = []


    #######Split the dataframe apart to impute the variables
    #Get datatypes for each column

    #split columns into two groups

    #impute according to user input

    #Bind back together


    ######Save transformations - TODO: how do I do this?
    return X #TODO: what do I return here?