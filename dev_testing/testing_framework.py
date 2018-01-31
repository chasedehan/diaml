#testing_framework.py

#This script is to evaluate an arbitrary number of classifier objects and output the results
#Build a class that takes in a model object and outputs a dataframe with the predictions

#Benefits of this approach are that we can initialize the class a single time, then feed different datasets in to test

import pandas as pd
from dev_testing.testing_funs import FitPipelines
from sklearn.datasets import load_boston, load_diabetes
from sklearn import linear_model
import lightgbm as lgb
from transformations.transformations import DiaPoly
from sklearn.pipeline import make_pipeline

dataset = load_boston()
X, y = dataset.data, dataset.target
X = pd.DataFrame(X)

gbm = lgb.LGBMRegressor()
lasso = linear_model.Lasso()
diaPoly = make_pipeline(DiaPoly(),
                        linear_model.Lasso())
diaPoly2 = make_pipeline(DiaPoly(replace=False),
                        linear_model.Lasso())
models = [lasso, gbm, diaPoly, diaPoly2]
tp = FitPipelines(n_folds=5, clf_list=models)
tp.fit(X, y)
# tp.results #results fromt he cv

#Now to evaluate the results

tp.eval_mse()
tp.model_mse



########################################################################################################################


#Want to show how the predictions from LightGBM and XGBoost are pretty closely related, but LASSO isn't
