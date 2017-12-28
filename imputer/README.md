#Imputer classes


There are two classes in this repo, they are:
* NewMissingColumn - creates a new column for each variable with NaN values above the cutoff threshold
* DataFrameImputer - imputes the most frequent value for categorical, user choice (mean, median, mode) for continuous.
  * This is implemented differently from sklearn's imputer because of the handling of categorical.
  * Idea borrowed from: [Stack Overflow user sveitser] (https://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn)
 
 
