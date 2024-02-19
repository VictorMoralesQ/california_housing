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

# Aqui hay que recoger la copia de la variable housing del archivo make_dataset.py
housing = pd.read_csv("/Users/vmxrls/Library/CloudStorage/Dropbox/Projects/california_housing/data/processed/train_set.csv")
# let's see how attributes are correlated with the median_house_value

# now we will try out attribute combinations
# for example, rooms_per_house, bedrooms_ratio or people_per_house
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

housing.to_csv("/Users/vmxrls/Library/CloudStorage/Dropbox/Projects/california_housing/data/processed/train_set.csv")

# and then we look at the correlation matrix again
# the new bedrooms_ratio is much more correlated that the number of rooms or bedrooms
# houses with less bedroom ratio tend to be more expensive
# the number of rooms per houshold is also more informative than the total number of rooms
# obviously the larger the houses, the more expensive they are
corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

