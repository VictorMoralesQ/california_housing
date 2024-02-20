
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from ..data.make_dataset import *



# now we are ready to select and train a machine learning model
# we start to train a very basic linear regression model
lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

housing_predictions = lin_reg.predict(housing)
print(housing_predictions[:5].round(-2))
print(housing_labels.iloc[:5].values)

# let's measure the performance of the model using RMSE on the whole training set
lin_rmse = mean_squared_error(housing_labels, housing_predictions,
                              squared=False)
print(lin_rmse)

# this is better than nothing but clearly not a great score
# the median housing value of most districts goes from 120k to 265k, so a typical prediction error of 68k is not really satisfying
# this is an example of underfitting the data, so we have to select a better model to train our dataset

# let's try with a DecisionTreeRegressor
tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)

# let's evaluate it on the training set
tree_reg.predict(housing)
tree_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
print(tree_rmse)

# We can see how the model has overfitted the data, so we need to use part of the training set for training and part of it for model validation
# we can do this using cross validation
tree_rmses = -cross_val_score(housing, housing_labels,
                              scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(tree_rmses).describe())

# we can see that the DecisionTreeRegressor performs as poorly as the Linear Regressor
# lastly, let's try to use a RandomForestRegressor
forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
                                scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(tree_rmses).describe())

# Let's compare this RMSE measured using cross-validation (the "validation error") with the RMSE measured on the training set (the "training error"):
forest_reg.fit(housing, housing_labels)
housing_predictions = forest_reg.predict(housing)
forest_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
print(forest_rmse)

# The training error is much lower than the validation error, which usually means that the model has overfit the training set. 
# Another possible explanation may be that there's a mismatch between the training data and the validation data, but it's not the case here, 
# since both came from the same dataset that we shuffled and split in two parts.                                                                                                                                                                                                                                                 