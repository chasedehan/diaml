# Stacking Models Module

This folder implements the stacking class of the semi-automated approach.  There are a number of ways to ensemble 
different learners - this is just one relatively straightforward way to incorporate the previous models into it. 

The objectives of this module are to:
* Take hyperparameters tuned within DiaML
* Take arbitrary learning models defined by user
    * As long as they conform to sklearn .fit() and .predict()
    
# Classes Implemented
* StackingAveragedModels() 
    * Takes in Level 0 base learners and Level 1 meta learner for stacking
* StackingModelsFeatureSelection()
    * Incorporates the BoostARoota feature selection algorithm to use different feature sets on each learner.
    
# Thanks
The classes included here are heavily inspired by Serigne's kernel on Kaggle:
https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

 
 
