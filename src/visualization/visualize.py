import matplotlib.pyplot as plt
import sys

sys.path.append('../features/')
from src.features import build_features
# we can also plot an histogram in order to see the type of data we are dealing with
build_features.housing.hist(bins=50, figsize=(12, 8))
plt.savefig("/reports/figures/describing_attributes.png")