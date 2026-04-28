import os
import numpy as np
import pandas as pd
import json 

# Ruta al CSV en la misma carpeta que este script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "tmdb_5000_movies.csv")
df = pd.read_csv(csv_path)


def obtener_generos(generos_str):
    data=json.loads(generos_str)
    return set(item["name"] for item in data)

df["genres"]=df["genres"].apply(obtener_generos)
#print(df["genres"])
#print(df[df["title"] == "The Matrix Revolutions"][["title","genres"]])
Matix_Generos = df[df["title"] == "The Matrix Revolutions"]["genres"].iloc[0]
print(Matix_Generos)





def jaccard_distance(A,B):
    return 1-len(A.intersection(B))/len(A.union(B))

distances = []
i=0
for _, row in df.iterrows():
    i+=1
    if row["title"] == "The Matrix Revolutions":
        continue
    TituloPeliculaActual = row["title"]
    GenerosPeliculaActual = row["genres"]
    #print("--------------------------------")
    #print(titulo)
    #print(generos)
    distance = jaccard_distance(Matix_Generos, GenerosPeliculaActual)
    distances.append((TituloPeliculaActual, distance, GenerosPeliculaActual))
    distances.sort(key=lambda x: (x[1], x[0]))
top10 = distances[:10]
top=[]
for titulo, distancia, generos in top10:
    print(f"{titulo}: {distancia:.3f}")

for distance in distances:
    if distance[1] == 0.0:
        top.append(distance[0])
print(len(top))

print("-------------Levenshtein Distance-------------------")
import numpy as np

def levenshtein_distance(s1, s2):
    rows = len(s1) + 1
    cols = len(s2) + 1
    
    dist_matrix = np.zeros((rows, cols), dtype=int)

    # Inicializar primera fila y columna
    for i in range(rows):
        dist_matrix[i, 0] = i
    for j in range(cols):
        dist_matrix[0, j] = j

    # Llenar matriz
    for i in range(1, rows):
        for j in range(1, cols):

            if s1[i-1] == s2[j-1]:
                cost = 0
            else:
                cost = 1

            dist_matrix[i, j] = min(
                dist_matrix[i-1, j] + 1,      # borrar
                dist_matrix[i, j-1] + 1,      # insertar
                dist_matrix[i-1, j-1] + cost  # sustituir
            )

    return dist_matrix[rows-1, cols-1]

Matrix_Overview=str(df[df["title"]=="The Matrix Revolutions"].iloc[0]["overview"])[:150]


# Acotar a 300 filas incluyendo la película de referencia (fila 125)
ref_iloc = 125
start = max(0, ref_iloc - 299)
end = min(len(df), start + 300)
if end - start < 300:
    start = max(0, end - 300)
df_lev = df.iloc[start:end]

distanceslevenshtein = []
for _, row in df_lev.iterrows():
    if row["title"] == "The Matrix Revolutions":
        continue
    title = row["title"]
    overview = str(row["overview"])[:150]
    distance = levenshtein_distance(Matrix_Overview, overview)
    distanceslevenshtein.append((title, distance))
distanceslevenshtein.sort(key=lambda x: (x[1], x[0]))
top10levenshtein = distanceslevenshtein[:10]
for titulo, distancia in top10levenshtein:
    print(f"{titulo}: {distancia:.3f}")
print("-----------------distancia eclidiana-------------------")

cols = ["budget", "revenue", "popularity"]
X = df[cols + ["title"]].copy()
X[cols] = (X[cols] - np.nanmean(X[cols], axis=0)) / np.nanstd(X[cols], axis=0, ddof=0)
matrix_title = "The Matrix Revolutions"
i_matrix = df.index[df["title"] == matrix_title][0]
v_matrix = X.loc[i_matrix, cols].values
distanceseuclidiana = []
for _, row in X.iterrows():
    if row["title"] == "The Matrix Revolutions":
        continue
    v = row[cols].values
    titulo = row["title"]
    distance = np.linalg.norm(v - v_matrix)
    distanceseuclidiana.append((titulo, distance))
distanceseuclidiana.sort(key=lambda x: (x[1], x[0]))
top10euclidiana = distanceseuclidiana[:3]
for titulo, distancia in top10euclidiana:
    print(f"{titulo}: {distancia:.3f}")





   
