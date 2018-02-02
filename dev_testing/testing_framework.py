#testing_framework.py

#This script is to evaluate an arbitrary number of classifier objects and output the results
#Build a class that takes in a model object and outputs a dataframe with the predictions

#Benefits of this approach are that we can initialize the class a single time, then feed different datasets in to test

import pandas as pd
import numpy as np
from dev_testing.testing_funs import FitPipelines
from sklearn.datasets import load_boston, load_diabetes
from sklearn import linear_model
import lightgbm as lgb
from transformations.transformations import DiaPoly
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

dataset = load_diabetes()
X, y = dataset.data, dataset.target
X = pd.DataFrame(X)
X.columns = ['Var_'+str(i) for i in X.columns]



gbm = lgb.LGBMRegressor()
lasso = linear_model.Lasso()
ridge = linear_model.Ridge()
alphas = np.logspace(-4, -0.5, 30)
diaPoly = make_pipeline(DiaPoly(),
                        linear_model.Ridge())
diaPoly2 = make_pipeline(DiaPoly(replace=False,subset_col=False),
                        linear_model.LassoCV(alphas=alphas, max_iter=3000))
rf = RandomForestRegressor()
# diaPoly.fit(X,y)


models = [gbm, rf, ridge, diaPoly]
tp = FitPipelines(n_folds=3, clf_list=[ridge,diaPoly])
tp.fit(X, y)
tp.results #results from the cv

#Now to evaluate the results
tp.eval_mse()



df = tp.results.copy()
df['Avg1'] = (df.Model_1 + df.Model_3) / 2
eval_mse(df)


