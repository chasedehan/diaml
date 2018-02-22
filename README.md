# DiaML
Does It All Machine Learning - DiaML automates the machine learning pipeline.

# Goals of Project
The stated goal of this project is to create a modular and automated machine learning process to take a dataset from start to completed model. This is a modular approach to allow a Data Scientist the ability to use certain components in their modeling process/pipeline without being forced into a one-size-fits-all box.  In this way, it allows the Data Scientist to strealime the process while still being able to customize certain elements.

# Existing Packages?
There are a number of packages in existence that are similar and well constructed.  DiaML differs from the below in the following ways:
* Scikit Learn - sklearn is a deep codebase allowing a Data Scientist to do almost anything they want out of the box.  DiaML heavily relies on sklearn methods, extending many of them to accomplish the goals of acclerating the model development process
* AutoML and Auto-WEKA - These algorithms focus exclusively on hyperparameter tuning and selecting the learning algorithm. DiaML incorporates some of this, but the focus is on many of the steps to get data ready for inserting into the last step of tuning and ensembling.
* TPOT - This package is attempting to do many of the same things DiaML attempts to solve, but approaches the problem differently.  TPOT aims to create the entire pipeline, then writing a new python script which can then be modified by the Data Scientist. DiaML seeks to solve this by implementing the classes on the front end rather than modifying the code after the algorithm has been run.

# Approach (Steps)
Below are the independent modules to be called depending on the users preferences.  They are independent and can be used as a single piece in the entire pipeline.  There is no obligation to use all of the methods.
1. Check dtypes, attempting to coerce any categorically coded variables into numeric
2. Impute Missing Variables, also creating variables "IsMissing" (Mostly Complete)
3. Outlier Detection and modification
4. Feature transformations for a more linear representation (Mostly Complete)
5. Feature Selection - I already built a tool, BoostARoota to fill in some of this
6. Hyperparameter Tuning of top X models
7. Stacking models - this is a distinct step from (6) in that it doesn't limit the Data Scientist from passing in models (with parameters) they deem to have high value.  Essentially, the Data Scientist can pass in (6) plus any other models desired as long as it has a .fit() and .predict() method associated.

# Installation
Easiest way is to use `pip`:
```
$ pip install diaml
```

## Usage  

This module is built to conform to sklearn's `fit()` and `transform()` methods.  Currently available classes are:  
* `NewMissingColumn()`: creates a new column for each variable with NaN values above the cutoff threshold
* `DataFrameImputer()`: imputes the most frequent value or user specified string for categorical, user choice (mean, median, mode) for continuous.
  * This is implemented differently from sklearn's imputer because of the handling of categorical missing values
* `DiaImputer()`: Executes NewMissingColumn and DataFrameImputer in a single call
* `DiaPoly()`: Automatically transforms data into a more linear representation through polynomial interpolation
  * This is probably the largest value add of the methods currently available


To use the imputer and the polynomial interpolation you can run the following, assuming you have X and y already split:
```python  
from diaml import DiaImputer, DiaPoly

# Below imputes continuous vars with "mode, while categorical variables are assigned a value of "missing_value"
dfi = DataFrameImputer(cont_impute="mode", cat_impute="missing_value", missing_value="missing_value")
new_X = dfi.fit_transform(X) # passing y is not necessary, but you can

dp = DiaPoly()
new_X = dp.fit_transform(new_X, y)
```


Alternatively, can place these inside of an sklearn Pipeline:
```python
from diaml import DiaImputer, DiaPoly
from sklearn.pipeline import Pipeline


dia_pipeline = Pipeline([('impute', DataFrameImputer()),
                             ('poly', DiaPoly())])

new_X = dia_pipeline.fit_transform(X,y)                          
```


## Want to Contribute?

This project has found some initial successes and there are a number of directions it can head.  It would be great to have some additional help if you are willing/able.  Whether it is directly contributing to the codebase or just giving some ideas, any help is appreciated.  The goal is to make the algorithms as robust as possible.  The primary focus right now is on the components under Future Implementations, but are in active development.  Please reach out to see if there is anything you would like to contribute in that part to make sure we aren't duplicating work.  


A special thanks to [Progressive Leasing](http://progleasing.com) for sponsoring this research.
