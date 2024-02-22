from ..data.make_dataset import *
from ..models.train_model import *
from scipy import stats
import numpy as np
import joblib

X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)

final_rmse = mean_squared_error(y_test, final_predictions, squared=False)

# we want know how precise is this estimation
# we can compute 95% confidence interval for the generalization error
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
interval_confidence = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))
print(interval_confidence)

# now we are ready to launch the project
joblib.dump(final_model, "my_california_housing_model.pkl")