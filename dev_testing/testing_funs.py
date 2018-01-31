#Testing_funs.py


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

class FitPipelines(object):
    def __init__(self, n_folds, clf_list):
        self.folds = n_folds
        self.clf_list = clf_list

    def fit(self, X, y):

        kf = KFold(n_splits=self.folds)

        for this_fold , (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            df_y = pd.DataFrame({'Fold': this_fold, 'y': y_test})

            for i, clf in enumerate(self.clf_list):
                clf.fit(X_train, y_train)
                df_y["Model_"+str(i)] = clf.predict(X_test)

            if 'results' not in locals():
                results = df_y.copy()
            else:
                results = results.append(df_y.copy(), ignore_index=True)
        self.results = results

    def eval_mse(self):
        df = self.results.copy()
        y_test = df['y']
        df = df.drop(['y', 'Fold'], axis=1)
        columns = df.columns
        model_mse = {}
        for col in columns:
            model_mse[col] = mean_squared_error(y_test, df[col])
        self.model_mse = model_mse
        print(self.model_mse)



class EvalPipelines(object):
    #Requires Dataframe in the form of a FitPipelines().results object
    def __init__(self):
        return self

    def evaluate(self):
        return self
