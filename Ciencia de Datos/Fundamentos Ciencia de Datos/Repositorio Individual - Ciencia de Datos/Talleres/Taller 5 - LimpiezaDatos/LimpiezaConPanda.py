import pandas as pd
from pathlib import Path

csv_path = Path(__file__).with_name("movies.csv")
df = pd.read_csv(csv_path)

df_limpio = df.copy()
#print(df_limpio.isna().sum())
nan_genre = df_limpio[df_limpio["GENRE"].isna()]
#print(nan_genre)
i=0
for index, row in nan_genre.iterrows():
    print(f"row: {row}")
    i+=1
    if i > 0:
        break

