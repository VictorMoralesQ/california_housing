from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

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
