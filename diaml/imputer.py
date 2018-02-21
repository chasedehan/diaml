import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
import warnings
import scipy.stats as ss

########################################################################################
#
# DiaImputer to combine the other methods in this module
#
########################################################################################

class DiaImputer(TransformerMixin):

    def __init__(self, cutoff=0.02, cont_impute=None, cat_impute=None, missing_value=None):
        self.cutoff = cutoff
        self.cont_impute = cont_impute
        self.missing_value = missing_value
        self.cat_impute = cat_impute

    def fit(self, X, y=None):
        self.new_cols = NewMissingColumn(cutoff=self.cutoff).fit(X)
        imputer_construct_dict = {}
        if self.cont_impute is not None: imputer_construct_dict['cont_impute'] = self.cont_impute
        if self.missing_value is not None: imputer_construct_dict['missing_value'] = self.missing_value
        if self.cat_impute is not None: imputer_construct_dict['cat_impute'] = self.cat_impute
        self.imputer = DataFrameImputer(**imputer_construct_dict).fit(X)
        return self

    def transform(self, X):
        new_X = self.new_cols.transform(X)
        new_X = self.imputer.transform(new_X)
        return new_X



########################################################################################
#
# Create New Variables for columns with missing values above threshold
#
########################################################################################


class NewMissingColumn(TransformerMixin):

    def __init__(self, cutoff=0.02):
        self.cutoff = cutoff

        # Throw errors if the inputted parameters don't meet the necessary criteria
        if (cutoff < 0) | (cutoff >= 1):
            raise ValueError('cutoff ' + str(cutoff) + ' is not in the range [0, 1)')

    def fit(self, X, y=None):
        #Check if there are any NA values
        if X.isnull().values.any():
            self._new_column_id(X)
        else:
            warnings.warn("No NA values in dataframe. Process Finished")
        return self

    def transform(self, X):
        #call new_missing_columns
        new_columns = [j + "_IsMissing" for j in self.new_missing_columns]
        for i in range(len(new_columns)):
            this_column = self.new_missing_columns[i]
            this_new_column = new_columns[i]
            #Create new column by coercing old boolean column into 1/0
            X[this_new_column] = X[this_column].isnull() * 1
        #call ohe if required
        return X

    def _new_column_id(self, X):
        nrow = X.shape[1]
        #Find the missing columns
        missing_columns = X.columns[X.isnull().any()]

        # if above threshold, create new columns
        self.new_missing_columns = []
        for i in missing_columns:
            if (sum(X[i].isnull()) / nrow) > self.cutoff:
                self.new_missing_columns.append(i)
        return self


########################################################################################
#
# Impute values - median for categorical, user choice for continuous
#
########################################################################################
#https://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn


class DataFrameImputer(TransformerMixin):

    def __init__(self, cont_impute='median', cat_impute='mode', missing_value='missing'):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value in column.
        Columns of other types are imputed with user selected (median as default) of column.
        """
        try:
            self.missing_value = str(missing_value)
        except ValueError:
            print("must enter missing_value parameter as a string")

        if cont_impute not in ["mean", "median", "mode", "missing_value"]:
            raise ValueError(str(cont_impute) + " is not a valid value for cont_impute. Must enter one of the following: ('mean', 'median', 'mode')")

        if cat_impute not in ['mode', 'missing_value']:
            raise ValueError(str(cat_impute) + " is not a valid value for cat_impute. Must enter one of the following: ('mode', 'missing_value')")

        self.cont_impute = cont_impute
        self.cat_impute = cat_impute

    def fit(self, X, y=None):
        def set_fill(self, X):
            self.fill = pd.Series([X[c].value_counts().index[0]
                                   if X[c].dtype == np.dtype('O') else self.this_fun(X[c]) for c in X],
                                  index=X.columns)

        def fill_cat_value(self, X):
            self.fill = pd.Series([self.missing_value
                                   if X[c].dtype == np.dtype('O') else self.this_fun(X[c]) for c in X],
                                  index=X.columns)

        #Set the imputer desired for each of the imputer values desired

        if self.cont_impute == 'mode':
            holder_fun = getattr(ss, 'mode')
            self.this_fun = lambda x: holder_fun(x)[0][0] #This is weird, but it works
            if self.cat_impute == 'missing_value':
                fill_cat_value(self, X)
            else:
                set_fill(self, X)
        else:
            self.this_fun = getattr(np, self.cont_impute)
            if self.cont_impute == 'median':
                self.fill = pd.Series([X[c].value_counts().index[0]
                                  if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
                                  index=X.columns)
            else:
                if self.cat_impute == 'missing_value':
                    fill_cat_value(self, X)
                else:
                    set_fill(self, X)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

