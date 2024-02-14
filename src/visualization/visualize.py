import matplotlib.pyplot as plt
import sys
import pandas as pd
from pandas.plotting import scatter_matrix

sys.path.append("/Users/vmxrls/Library/CloudStorage/Dropbox/Projects/california_housing/src/features/build_features.py")
# we can also plot an histogram in order to see the type of data we are dealing with
housing = pd.read_csv("/Users/vmxrls/Library/CloudStorage/Dropbox/Projects/california_housing/data/raw/housing.csv")
def plot_hist_features(housing):
    housing.hist(bins=50, figsize=(12, 8))
    plt.savefig("/Users/vmxrls/Library/CloudStorage/Dropbox/Projects/california_housing/reports/figures/describing_features.png")

plot_hist_features(housing)

housing = pd.read_csv("/Users/vmxrls/Library/CloudStorage/Dropbox/Projects/california_housing/data/interim/housing.csv")
# Ploting number of districts per each category based in the column income_cat
# created previously
def plot_bar_income_cat(housing):
    housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
    plt.xlabel("Income category")
    plt.ylabel("Number of districts")
    plt.savefig("/Users/vmxrls/Library/CloudStorage/Dropbox/Projects/california_housing/reports/figures/income_cat_hist.png")

plot_bar_income_cat(housing)

# because the dataset includes geographical data
# it's a good idea to create a scatterplot of all the districts to visualize data
housing = pd.read_csv("/Users/vmxrls/Library/CloudStorage/Dropbox/Projects/california_housing/data/processed/train_set.csv")
def california_map(housing):
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
    plt.savefig("/Users/vmxrls/Library/CloudStorage/Dropbox/Projects/california_housing/reports/figures/california_housing_prices_map.png")

california_map(housing)

# as we could see in the correlation matrix, there's a positive correlation between the house_median_value and the median_income
# we cna also see there's a negative correlation between the latitude and the house_median_value
# that means, prices have a slightly tendency to go down when we go north

# another way to check for correlation is to print a scatter_matrix
def scatter_matrix(housing):
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.savefig("/Users/vmxrls/Library/CloudStorage/Dropbox/Projects/california_housing/reports/figures/scatter_matrix.png")

scatter_matrix(housing)

# looking at the correlation scatterplots, the most promising value to predict the median house value is the median income
# the plot reveals the correlation is few strong because of the upward trend and the points not too dispersed
# there are lines around 500k, 450k and 280k, we want to remove the districts to prevent the algorithms to reproduce these data quirks
def median_income_corr(housing):
    housing.plot(kind="scatter", x="median_income", y="median_house_value",
                 alpha=0.1, grid=True)
    plt.savefig("/Users/vmxrls/Library/CloudStorage/Dropbox/Projects/california_housing/reports/figures/median_income_corr.png")

median_income_corr(housing)