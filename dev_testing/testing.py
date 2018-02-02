import pandas as pd
import numpy as np
import urllib


########################################################################################################################
#
#  Import Madelon Dataset
#
########################################################################################################################
train_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data'
# download the file
raw_data = urllib.request.urlopen(train_url)
train = pd.read_csv(raw_data, delim_whitespace=True, header=None)
train.columns = ["Var"+str(x) for x in range(len(train.columns))]
train = pd.get_dummies(train)
labels_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels'
raw_data = urllib.request.urlopen(labels_url)
labels = pd.read_csv(raw_data, delimiter=",", header=None)
labels.columns = ["Y"]
labels = (labels + 1) /2
# labels = np.array(labels, dtype=int)
labels = labels.Y.values



train[['Var1']] = train[['Var1']].astype(str)
train.iloc[1:4,1:10] = np.NaN
########################################################################################################################
#
#  Test that imputer is working
#
########################################################################################################################
from imputer.imputer import NewMissingColumn, DataFrameImputer, DiaImputer

new_cols = NewMissingColumn(cutoff=0)
new_cols.fit(train)
new_X = new_cols.transform(train)
new_X.shape

#impute values according to user specifications
dfi = DataFrameImputer(cont_impute="mode", cat_impute="missing_value", missing_value="-9999999")
dfi.fit(train)
new_X = dfi.transform(train)
new_X.head()

#So, stringing these together we see:
from sklearn.pipeline import Pipeline
missing_pipeline = Pipeline([('new_cols', NewMissingColumn(cutoff=0)),
                             ('impute', DataFrameImputer(cont_impute="mean"))])
new_X = missing_pipeline.fit_transform(train)
new_X.shape

#Do both of the above with a single call
DI = DiaImputer(cont_impute="mode")
DI.fit(train)
new_X = DI.transform(train)
new_X.shape


########################################################################################################################
#
#  work out the linear transformations
#
########################################################################################################################
#Below is the general structure of how the
from transformations.transformations import DiaLog, DiaPoly, DiaSkew


#Id and transform
dl = DiaLog()
new_X = dl.fit_transform(train)

dp = DiaPoly()
new_X = dp.fit_transform(train)

ds = DiaSkew()
new_X = ds.fit_transform(train)


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
new_X = enc.fit_transform(train)


########################################################################################################################
#
#  General testing framework to check the parformance
#
########################################################################################################################
from sklearn.model_selection import KFold
import lightgbm as lgb

class TestThis(object):
    def __init__(self, clf, compare_class, folds=5):
        self.clf = clf
        self.compare_class = compare_class
        self.folds = folds

    def fit(self, X, y):
        #Cre# ate folds

        kf = KFold(n_splits=self.folds)
        self.raw_y_hat = []
        self.trans_y_hat = []
        self.lgb_y_hat = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.clf.fit(X_train, y_train)
            self.raw_y_hat.extend(self.clf.predict(X_test))

            gbm = lgb.LGBMRegressor()
            gbm.fit(X_train, y_train)
            self.lgb_y_hat.extend(gbm.predict(X_test))

            self.compare_class.fit(X_train, y_train)
            new_X = self.compare_class.transform(X_train)
            new_test_X = self.compare_class.transform(X_test)

            self.clf.fit(new_X, y_train)
            self.trans_y_hat.extend(self.clf.predict(new_test_X))


from sklearn.datasets import load_boston, load_diabetes
from sklearn import linear_model
dataset = load_diabetes()
X, y = dataset.data, dataset.target
X = pd.DataFrame(X)

lasso = linear_model.Lasso()
ridge = linear_model.Ridge()
dp = DiaPoly()

testing = TestThis(clf=lasso, compare_class=dp)
testing.fit(X,y)
from sklearn.metrics import mean_squared_error
print("MSE raw data lasso " + str(mean_squared_error(testing.raw_y_hat, y)))
print("MSE transformed lasso " + str(mean_squared_error(testing.trans_y_hat, y)))
print("MSE raw data lightGBM " + str(mean_squared_error(testing.lgb_y_hat, y)))


########################################################################################################################
#
#  Check on a single holdout
#
########################################################################################################################

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import lightgbm as lgb

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#transformed
dp = DiaPoly()
dp.fit(X_train,y_train)
new_X = dp.transform(X_train)
new_X_test = dp.transform(X_test)
ridge = linear_model.Ridge()
ridge.fit(new_X,y_train)
y_hat_ridge = ridge.predict(new_X_test)


#lightGBM
gbm = lgb.LGBMRegressor()
gbm.fit(X_train,y_train)
y_hat_gbm = gbm.predict(X_test)

#lightGBM with dp
gbm.fit(new_X, y_train)
y_hat_gbm_dp = gbm.predict(new_X_test)

#Basic Ridge
ridge = linear_model.Ridge()
ridge.fit(X_train,y_train)
y_hat_ridge_all = ridge.predict(X_test)

#Random Forest
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
y_hat_rf = rf.predict(X_test)
#Random Forest with dp
rf.fit(new_X,y_train)
y_hat_rf_dp = rf.predict(new_X_test)

#Try PolynomialFeatures Again:
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                   ('ridge', linear_model.Ridge())])
model.fit(new_X, y_train)
y_hat_poly = model.predict(new_X_test)

print("MSE raw ridge " + str(mean_squared_error(y_hat_ridge_all, y_test)))
print("MSE transformed ridge " + str(mean_squared_error(y_hat_ridge, y_test)))
print("MSE raw data lightGBM " + str(mean_squared_error(y_hat_gbm, y_test)))
print("MSE transformed lightGBM " + str(mean_squared_error(y_hat_gbm_dp, y_test)))
print("MSE raw data RandomForest " + str(mean_squared_error(y_hat_rf, y_test)))
print("MSE transformed RandomForest " + str(mean_squared_error(y_hat_rf_dp, y_test)))
print("MSE transformed polyFeatures " + str(mean_squared_error(y_hat_poly, y_test)))






########################################################################################################################
#
#  Test stacking
#
########################################################################################################################
from stacking.stacking import StackingAveragedModels
GBoost = GradientBoostingClassifier()
GBoost.fit(train, labels)
GBoost.predict(train)

stacker = StackingAveragedModels(regression=False)
stacker.fit(train, labels)

KRR = KernelRidge()
ENet = ElasticNet()
stacker = StackingAveragedModels(regression=False)
stacker.fit(train, labels)
stacker.predict(train)