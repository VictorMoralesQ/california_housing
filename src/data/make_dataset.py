from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
from sklearn.model_selection import train_test_split

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
train_set.to_csv("/Users/vmxrls/Library/CloudStorage/Dropbox/Projects/california_housing/data/processed/train_set.csv")
test_set.to_csv("/Users/vmxrls/Library/CloudStorage/Dropbox/Projects/california_housing/data/processed/test_set.csv")

# since we are going to experiment with various transformations on the full training set
# we make a copy of the original set 
housing = train_set.copy()