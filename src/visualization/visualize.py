import matplotlib.pyplot as plt
import sys
import pandas as pd

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