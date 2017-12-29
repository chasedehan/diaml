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


train.iloc[1:4,1:10] = np.NaN
########################################################################################################################
#
#  Test that imputer is working
#
########################################################################################################################
from imputer.imputer import NewMissingColumn, DataFrameImputer

new_cols = NewMissingColumn(cutoff=0)
new_cols.fit(train)
new_X = new_cols.transform(train)
new_X.shape

#impute values according to user specifications
dfi = DataFrameImputer(cont_impute="mode")
dfi.fit(train)
new_X = dfi.transform(train)

#So, stringing these together we see:
from sklearn.pipeline import Pipeline
missing_pipeline = Pipeline([('new_cols', NewMissingColumn(cutoff=0)),
                             ('impute', DataFrameImputer(cont_impute="mean"))])
new_X = missing_pipeline.fit_transform(train)
new_X.shape

DI = DiaImputer()
DI.fit(train)
new_X = DI.transform(train)
new_X.shape



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