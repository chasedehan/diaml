# Imputer classes

## Purpose

There are two main classes, with one wrapper around the two:
* NewMissingColumn - creates a new column for each variable with NaN values above the cutoff threshold
* DataFrameImputer - imputes the most frequent value or user specified string for categorical, user choice (mean, median, mode) for continuous.
  * This is implemented differently from sklearn's imputer because of the handling of categorical missing values
* DiaImputer - Executes NewMissingColumn and DataFrameImputer in a single call


## Usage
This module can be implemented in a similar manner to sklearns `fit()` and `transform()` and can be used in pipelines. 
All classes in this module require data to be passed in as a pandas dataframe.  


### NewMissingColumn
We can create a new column for any existing column in X that has missing values greater than the cutoff.  The default cutoff
is 0.02 which means that at least 2% of the observations in the dataframe must have NaN values in order 
to create a new column.  In the below example, we have set the cutoff to 0.05.
```python
new_cols = NewMissingColumn(cutoff=0.05)
new_cols.fit(X)
new_X = new_cols.transform(X)
#or
new_X = new_cols.fit_transform(X)
```

### DataFrameImputer
Instead of inserting the same value into every place there is a missing value, we can specify the types of values we
would like to insert.  The defaults are to impute continuous variables with the median and categorical variables with 
the most frequent.  
```python
dfi = DataFrameImputer(cont_impute="mean")
dfi.fit(X)
new_X = dfi.transform(X)
```

There are two ways in this module to impute the missing values in categorical variables.  The first (shown above) is
to impute the most frequent, while the second is shown immediately below.  We can use the arguments 'cat_impute' and
'missing_value' to insert another category in the same column indicating that the value is missing.  The default 'missing_value'
is 'missing'.
```python
dfi = DataFrameImputer(cont_impute="mode", cat_impute="missing_value", missing_value="-9999999")
dfi.fit(train)
new_X = dfi.transform(train)
```

### Stringing Methods Together
There are a couple ways to string these together.  One of which is to use sklearn's pipelines; the other implemented
here is to use DiaImputer().  Both approaches are shown below and deliver identical results.

```python
#Using the pipeline from sklearn
from sklearn.pipeline import Pipeline
missing_pipeline = Pipeline([('new_cols', NewMissingColumn(cutoff=0)),
                             ('impute', DataFrameImputer(cont_impute="mean", cat_impute="missing_value"))])
new_X = missing_pipeline.fit_transform(train)
  
#Using the wrapper functionality in DiaML
DI = DiaImputer(cutoff=0, cont_impute="mean", cat_impute="missing_value")
new_X = DI.fit_transform(train)
```



 
