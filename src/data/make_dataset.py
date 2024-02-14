from pathlib import Path
import pandas as pd
import numpy as np
import tarfile
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

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