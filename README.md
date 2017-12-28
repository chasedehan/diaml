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
2. Impute Missing Variables, also creating variables "IsMissing"
3. Outlier Detection and modification
4. Feature transformations for a more linear representation
5. Feature Selection - I already built a tool, BoostARoota to fill in some of this
6. Hyperparameter Tuning of top X models
7. Stacking models - this is a distinct step from (6) in that it doesn't limit the Data Scientist from passing in models (with parameters) they deem to have high value.  Essentially, the Data Scientist can pass in (6) plus any other models desired as long as it has a .fit() and .predict() method associated.

# Installation
Eventually, this will on PyPi, but it is still too rapidly being developed.  For now, just take a look around.



