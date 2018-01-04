#transformations.py

#The objective here is to transform a certain subset of variables for a more linear representation.


# Make log transformations
from sklearn.preprocessing import FunctionTransformer
    # we use this because it has a transform method already associated with it


# Polynomial Features
from sklearn.preprocessing import PolynomialFeatures

#For the polynomials, can use Polynomial Interpolation by running through a Lasso and see what sticks
#http://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html#sphx-glr-auto-examples-linear-model-plot-polynomial-interpolation-py

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#TODO: clean up the below to make it all roll together
# for count, degree in enumerate([3, 4, 5]):
#     #How do I accumulate the results from each of these
#     model = make_pipeline(PolynomialFeatures(degree), Ridge())
#     model.fit(X, y)
#     y_plot = model.predict(X_plot)


#####Checking distributions

#Skewness - determining the variables to apply Box Cox to
#identify the variables to transform
from scipy.stats import skew
skewness = skew(data)
#Split out those above threshold
transform = skewness > threshold
#applying box cox to certain variables:
from scipy.stats import boxcox
boxcox_data = boxcox(transform)
#Put back together
cbind(boxcox_data, non_boxcox)

#Center-Scale - this is necessary in many regularized models
    # Demean and look like normally distributed data
from sklearn import preprocessing
preprocessing.scale(X_train) #Does this have a fit/transform?

#Can scale features according to a range, i.e. 0-1
    #Why? preserves zero entries in sparse data
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)


#PCA transformations
#It is not enough to center scale independently, can use PCA to remove linear dependence

#http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html

#Target Variable transformation
    #How do I automate this transformation?
    #One thought is to create a bunch of different transformations on Y, then run a simple lasso, comparing the R2
        #Then, keep the best representation
        #Only want to do this on regression tasks

    #Also have scale() and StandardScalar() to scale the target variable





