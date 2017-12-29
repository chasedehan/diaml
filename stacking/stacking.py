from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, Ridge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import itertools

import pandas as pd
import numpy as np
#Thanks: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

def _set_models(regression=True):
    #TODO: need to allow user to pass in how many models they want to pass in
    if regression:
        GBoost = GradientBoostingRegressor()
        # model_xgb = xgb.XGBRegressor()
        model_lgb = lgb.LGBMRegressor()
        KRR = KernelRidge()
        ENet = ElasticNet()
        lasso = Lasso()
        return (GBoost, model_lgb, KRR, ENet, lasso)
    else:
        GBoost = GradientBoostingClassifier()
        # model_xgb = xgb.XGBClassifier()
        model_lgb = lgb.LGBMClassifier()
        KRR = KernelRidge()
        ENet = ElasticNet()
        lasso = Lasso()
        return (GBoost, model_lgb, KRR, ENet, lasso)

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, diaml_tuned=None, base_models=None, meta_model=None, n_folds=3, regression=True):
        if base_models is not None:
            self.base_models = base_models
        else:
            self.base_models = _set_models(regression=regression)
        if meta_model is not None:
            self.meta_model = meta_model
        else:
            self.meta_model = Lasso()
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)


        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                self.train_index = train_index
                self.X = X

                print(str(X.shape))
                print(str(len(train_index)))
                this_X = X.iloc[train_index].copy()
                this_y = y[train_index]
                print(type(this_X))
                print(len(this_y))
                # instance.fit(X[train_index], y[train_index])
                instance.fit(this_X, this_y)
                y_pred = instance.predict(X.iloc[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


# Incorporates model independent feature selection (i.e. one subset for RF, one for XGB, etc)
class StackingModelsFeatureSelection(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y): # Need to pass X as a pandas df
        # self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        #Set the params to be passed into the run_bar()
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        #Run feature selection on the various parameters
        print("Starting Feature Selection")
        bar_params = [{'max_rounds': 1},
                      {'max_rounds': 2},
                      {'max_rounds': 3},
                      {'delta':.05},
                      {'delta':.2},
                      {'cutoff':2},
                      {'cutoff':6},
                      {'cutoff': 10, 'max_rounds':1}]
        #Now extract the br features
        feature_list = []
        for bar_param in bar_params:
            feature_list.append(run_bar(X, y, bar_param))
        #Add back in all_features
        feature_list.append(X.columns.tolist())
        #Delete any duplicates in the feature lists
        feature_list.sort()
        feature_list = list(k for k, _ in itertools.groupby(feature_list))
        print("Feature Selection Completed")
        #TODO: Pass those feature subsets into the grid
        self.grid_ = [(a, b) for a in self.base_models for b in feature_list]
        self.base_models_ = [list() for x in self.grid_]
        # Train cloned base models then create out-of-fold predictions - for feeding into the meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.grid_)))
        #TODO: start the iteration through the following loop to extract the feature set and models
        for i, this_grid in enumerate(self.grid_):
            #todo: subset the data appropriately
            model = this_grid[0]
            these_features = this_grid[1]
            this_X = X[these_features].values
                #todo: change the X below inside the loop to this_X
            for train_index, holdout_index in kfold.split(this_X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(this_X[train_index], y[train_index])
                y_pred = instance.predict(this_X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        #todo: have to change the feature subsets in the predict so that it knows to change the feature set
        base_preds = []
        for i in range(len(self.grid_)):
            these_vars = self.grid_[i][1]
            this_X = X[these_vars]
            base_models = self.base_models_[i]
            base_preds.append(np.column_stack([model.predict(this_X) for model in base_models]).mean(axis=1))
        meta_features = np.column_stack(base_preds)
        return self.meta_model_.predict(meta_features)