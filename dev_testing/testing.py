import pandas as pd
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
labels_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels'
raw_data = urllib.request.urlopen(labels_url)
labels = pd.read_csv(raw_data, delimiter=",", header=None)
labels.columns = ["Y"]

########################################################################################################################
#
#  Test that imputer is working
#
########################################################################################################################

im = ImputeMissing()
im.fit(train)
new_x = im.transform(train)
#or
new_x = im.fit_transform(train)