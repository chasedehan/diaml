# cv_model.py
"""
Objective: build out the nested CV folds for the dialer model
Why: Hyper-parameter optimization and model selection
How: Create a class like a classifier you pass the whole dataset into

"""

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet, Lasso, Ridge
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from diaml import DiaPoly
from boostaroota import BoostARoota
import random

# TODO: do we work out the bayesian optimization or do we use a simple grid search?
# TODO: also do posterior weighting of base learners. How? NN? monte carlo weighting? etc?

class NestedCV(TransformerMixin):
    def __init__(self, parameter_tuning=False, classifier_list=None):
        self.results = None
        self.new_results = None

        if classifier_list is None:
            self.classifiers = [make_pipeline(DiaPoly(), lgb.LGBMClassifier()),
                                make_pipeline(DiaPoly(), LogisticRegression()),
                                make_pipeline(BoostARoota(metric='logloss'), lgb.LGBMClassifier()),
                                make_pipeline(BoostARoota(metric='logloss'), DiaPoly(), lgb.LGBMClassifier()),
                                # make_pipeline(BoostARoota(metric='logloss', max_rounds=1), lgb.LGBMClassifier()),
                                # make_pipeline(BoostARoota(metric='logloss', max_rounds=1), GradientBoostingClassifier()),
                                ElasticNet(),
                                LogisticRegression(),
                                Lasso(),
                                # Ridge(),
                                lgb.LGBMClassifier(),
                                CatBoostClassifier(iterations=100, logging_level='Silent'),
                                # make_pipeline(BoostARoota(metric='logloss'), CatBoostClassifier(iterations=1000, logging_level='Silent')),
                                CatBoostClassifier(iterations=1000, logging_level='Silent') ]

        else:
            self.classifiers = classifier_list

        # self.second_classifiers = [LogisticRegression(), Lasso()]

    def print_results(self, y):
        for column in self.results:
            list = []
            [list.append([x[1], 1 - x[1]]) for x in self.results[column].iteritems()]
            print("Model: {}, with log loss: {}".format(column, log_loss(y, list)))

    def _ensemble_weights(self):
        TOP_MODELS = 5
        N_WEIGHTS = 1000

        self.final_weights = []
        weighted_results = []
        for fold in self.results['Fold'].unique():
            results = self.results.loc[self.results['Fold'] != fold].copy()
            y = results["YTest"]
            del results["YTest"]
            del results["Fold"]
            # Extract the log loss for each clf and put into a dictionary
            ll_results = {}
            for column in results:
                list = []
                [list.append([x[1], 1 - x[1]]) for x in results[column].iteritems()]
                ll_results[column] = log_loss(y, list)

            # Sort that dictionary and extract the clf names to use
            sorted_logloss = [(k, ll_results[k]) for k in sorted(ll_results, key=ll_results.get)]
            top_clf = [x[0] for x in sorted_logloss[0:TOP_MODELS]]
            top_results = results[top_clf]

            # Weight the top clf and extract into
            new_results = pd.DataFrame()
            weight_dict = {}
            for i in range(N_WEIGHTS):
                # Generate weights and make sure they sum to 1
                these_weights = [random.randrange(-100, 100, 1) for _ in range(TOP_MODELS)]
                if sum(these_weights) == 0:
                    these_weights = [x / 0.0001 for x in these_weights]
                else:
                    these_weights = [x / sum(these_weights) for x in these_weights]

                weight_dict["W_" + str(i)] = these_weights
                new_results["W_" + str(i)] = (top_results * these_weights).sum(axis=1)

            # Get the top 10-20 weights and average them:
            ll_results = {}
            for column in new_results:
                list = []
                [list.append([x[1], 1 - x[1]]) for x in new_results[column].iteritems()]
                ll_results[column] = log_loss(y, list)

            sorted_logloss = [(k, ll_results[k]) for k in sorted(ll_results, key=ll_results.get)]
            top_weights = [x[0] for x in sorted_logloss[0:10]]
            top_weights = pd.DataFrame({k: v for (k, v) in weight_dict.items() if k in top_weights}).mean(axis=1)

            final_weights = {}
            for i in range(len(top_clf)):
                final_weights[top_clf[i]] = top_weights[i]

            self.final_weights.append(final_weights)
            # Then, weight the out of sample
            oos_results = self.results.loc[self.results['Fold'] == fold].copy()

            these_weights = [final_weights[k] for k in final_weights]
            holder = (oos_results[top_clf] * these_weights).sum(axis=1)
            weighted_results.extend(holder)
        self.results["Weighted"] = weighted_results

    def train(self, X, y):
        """
        Comes up with the optimal hyperparameters and the ensemble weights
        """
        # Should run some feature selection in the top here before dropping into the next section

        self.results = pd.DataFrame()

        kf = KFold(n_splits=3)
        fold = 1
        for train, test in kf.split(X):
            X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y[train], y[test]
            # Train the base learners
            this_fold_results = pd.DataFrame()
            this_fold_results["YTest"] = y_test
            this_fold_results["Fold"] = [fold] * X_test.shape[0]

            for i, classifier in enumerate(self.classifiers):
                clf = clone(classifier)
                clf.fit(X_train, y_train)
                try:
                    this_fold_results["clf_" + str(i)] = [x[0] for x in clf.predict_proba(X_test)]
                except:
                    this_fold_results["clf_" + str(i)] = 1 - clf.predict(X_test)

            self.results = self.results.append(this_fold_results)
            fold += 1
        self._ensemble_weights()



    def fit(self, X, y):
        if self.results is None:
            return "must train() model first"
        else:
            self.classifier_output = []
            for classifier in self.classifiers:
                clf = clone(classifier)
                self.classifier_output.append(clf.fit(X, y))

            # Random Weightings # TODO: should really check the best in the above section - monte carlo?


    def predict(self, X):
        if self.classifier_output is None:
            return "must fit model first"
        else:
            results = pd.DataFrame()
            for i, clf in enumerate(self.classifier_output):
                try:
                    results["clf_" + str(i)] = [x[0] for x in clf.predict_proba(X)]
                except:
                    results["clf_" + str(i)] = 1-clf.predict(X)

            for i, weights in enumerate(self.final_weights):
                these_clf = [*weights]
                these_weights = [weights[k] for k in weights]
                results["weighted_"+str(i)] = (results[these_clf] * these_weights).sum(axis=1)
            self.new_results = results
            return results

    # TODO: need to write out a classifier with only the relevant data (i.e. what is the best)
    # Or, some method to specify which you would like to use
    # then, def predict() would be accurate

    def compare_results(self, y_test):
        if self.new_results is None:
            return "must predict() first"
        else:
            for column in self.new_results:
                # WTF: breaking it back into a list for accurate results
                list = []
                [list.append([x[1], 1 - x[1]]) for x in self.new_results[column].iteritems()]
                print("Model: {}, with log loss: {}".format(column, log_loss(y_test, list)))







# TODO: pass in pipelines as options to train and compare against
class diaML(TransformerMixin):
    """
    This is the outer loop of the diaML class, should reference InnerLoop inside each fold
    train_eval: for evaluating the performance on the whole
    fit: fits model on all data with same clf as specified in train_eval
    transform: predicts with new data :todo: identify type of y
    """
    def __init__(self, n_folds=3, folds=None, eval_metric='logloss', clf=None, classifier_list=None, parameter_tuning=False):
        if folds is not None:   # allow user to pass in predefined folds
            self.folds = folds
        else:
            self.n_folds = n_folds

        self.parameter_tuning = parameter_tuning
        self.eval_metric=eval_metric # TODO: make this some smarter logic

        self.classifier_list = classifier_list
        #
        # if classifier_list is not None:
        #     self.classifier_list = classifier_list
        # else:
        #     self.classifier_list =
            # TODO: place pipelines as primary, versus checking the classifiers
            # Then, we are going to want to pass these pipelines or classifiers into the inner loop
            # Inside the inner loop, we do all evaluation and parameter tunings
            # Then, it returns the "optimal" classifier --> plus maybe some of the base learners?

        if clf is None:
            self.clf = NestedCV()
        else:
            self.clf = clf

    def train_eval(self, X, y):
        kf = KFold(n_splits=self.n_folds)
        fold = 1
        self.preds_df = pd.DataFrame()
        for train, test in kf.split(X):
            X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y[train], y[test]
            try:
                ncv = NestedCV(classifier_list=self.classifer_list)
            except:
                ncv = NestedCV()
            ncv.train(X_train, y_train)
            ncv.fit(X_train, y_train)
            self.preds_df = self.preds_df.append(ncv.predict(X_test))
            fold += 1

    def compare_results(self, y_test):
        if self.preds_df is None:
            return "must predict() first"
        else:
            for column in self.preds_df:
                # WTF: breaking it back into a list for accurate results
                list = []
                [list.append([x[1], 1 - x[1]]) for x in self.preds_df[column].iteritems()]
                print("Model: {}, with log loss: {}".format(column, log_loss(y_test, list)))


    def fit(self, X, y):
        #Check the name, train only the models that show promise
        self.clf.fit(X, y)

    def transform(self, X):
        return self.clf.predict_proba(X)
