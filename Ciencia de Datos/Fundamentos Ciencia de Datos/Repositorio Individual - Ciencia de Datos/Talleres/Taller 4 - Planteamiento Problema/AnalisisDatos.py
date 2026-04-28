import pandas as pd
from pathlib import Path
datos = pd.read_csv(Path(__file__).with_name("insurance_claims.csv"))
print(datos.head())
print(datos.info())
print(datos.describe())
