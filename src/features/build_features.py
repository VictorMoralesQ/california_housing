import pandas as pd
import numpy as np

# read and save the dataset as a pandas dataframe in a varibale called 'housing'
housing = pd.read_csv("/Users/vmxrls/Library/CloudStorage/Dropbox/Projects/california_housing/data/raw/housing.csv")
housing.head()

# now we can see a quick description of the data such as n_rows, attributes' type and n_non-null values
housing.info()

# we see that the attribute 'total_bedrooms' contains some null values
# moreover, we see 'ocean_proximity' attribute is an object
# and we know that it must be a text attribute
# if we look the first five rows of the dataset, we can find out that is a categorical attribute
housing["ocean_proximity"].value_counts()

# let's look at the other fields using the describe() methdod
housing.describe()

# Creating a new category attribute with five categories to categorize 
# ranges from the median_income column
housing["income_cat"] = pd.cut(housing["median_income"], 
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
housing = housing.to_csv("/Users/vmxrls/Library/CloudStorage/Dropbox/Projects/california_housing/data/interim/housing.csv")