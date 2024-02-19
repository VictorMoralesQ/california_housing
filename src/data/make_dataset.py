from pathlib import Path
import pandas as pd
import numpy as np
import tarfile
import urllib.request
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

def load_housing_data():
    """Esta función descarga y carga los datos de vivienda desde un archivo tarball remoto.

    Returns:
        DataFrame: Un DataFrame de pandas que contiene los datos de vivienda.
    """
    # Ruta del directorio donde se encuentra el script make_dataset.py
    current_directory = Path(__file__).resolve().parent

    # Rutas de los archivos y carpetas
    tarball_path = current_directory / "../../data/raw/housing.tgz"
    csv_file_path = current_directory / "../../data/raw/housing/housing.csv"

    # Descarga y extrae el archivo tarball si no existe
    if not tarball_path.is_file():
        tarball_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path=tarball_path.parent)

    # Lee el archivo CSV
    housing = pd.read_csv(csv_file_path)

    return housing

housing = load_housing_data()

# Now, let's split the dataset into a training set and a test set
housing = pd.read_csv("/Users/vmxrls/Library/CloudStorage/Dropbox/Projects/california_housing/data/interim/housing.csv")
train_set , test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"], 
                                        random_state=42)

# since we are going to experiment with various transformations on the full training set
# we make a copy of the original set 
housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()

# we also need to take care of missing values
# in our case, we will use SimpleImputer to set the median in all the missing values of each column
imputer = SimpleImputer(strategy="median")

# since the median can only be computed on numerical attributes, we need to create a copy of the set
# with only the numerical attributes
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
X = imputer.transform(housing_num)

# wrapping X in a DataFrame recovering the column names and index from housing_num
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)
housing.to_csv("/Users/vmxrls/Library/CloudStorage/Dropbox/Projects/california_housing/data/interim/housing_tr.csv")

# now we deal with categorical attributes
housing_cat = housing[["ocean_proximity"]]

# let's comvert these categorical attributes from text to numbers
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

# in this context, each attibute is independent to the others
# so we create a binary attribute per category
cat_encoder = OneHotEncoder(sparse_output=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

df_test_unknown = pd.DataFrame({"ocean_proximity": ["<2H OCEAN", "ISLAND"]})
pd.get_dummies(df_test_unknown)

cat_encoder.handle_unknown = "ignore"
cat_encoder.transform(df_test_unknown)


df_output = pd.DataFrame(cat_encoder.transform(df_test_unknown),
                         columns=cat_encoder.get_feature_names_out(),
                         index=df_test_unknown.index)
print(df_output)

housing.to_csv("/Users/vmxrls/Library/CloudStorage/Dropbox/Projects/california_housing/data/processed/train_set.csv")
test_set.to_csv("/Users/vmxrls/Library/CloudStorage/Dropbox/Projects/california_housing/data/processed/test_set.csv")

# scaling features
# option 1
# using min-max scaler, the values are shifted and rescaled so they end up ranging from 0 to 1.
min_max_scaler = MinMaxScaler()
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

# option 2
# we can scale the values using standarization, being much less afected by possible outliers
std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

# custom transformers
# we transform features with heavy-tailed distributions by replcing them with thwir logarithm
log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

# now we build a pipeline to preprocess the numerical values
num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                             StandardScaler())

housing_num_prepared = num_pipeline.fit_transform(housing_num)

# we also can create a single transfomer capabling to handle all columns
num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms", 
               "total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"),
                             OneHotEncoder(handle_unknown="ignore"))

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])
# lastly, we build the pipeline to do all the transformations we need
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]
def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]
def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler()
    )
log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler()
)
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())
preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)

housing_prepared = preprocessing.fit_transform(housing)
print(preprocessing.get_feature_names_out())
