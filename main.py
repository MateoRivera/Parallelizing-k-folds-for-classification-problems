# Librerías para manipular datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Librería para medir el tiempo
import time

# Librería para graficar
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Librería para dividir el dataset
from sklearn.model_selection import train_test_split

# Librería propia para generar los experimentos
from experiments import experiments_knn

if __name__ == "__main__":
    # Se carga el dataset
    df = pd.read_csv(
        "https://raw.githubusercontent.com/dbdmg/data-science-lab/master/datasets/mnist_test.csv",
        header=None
        )
    
    # Se separan las características de la clase
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    # Se separan los datos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Se escalan los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Se experimenta con diferentes valores de k
    ks = [i for i in range(1, 500, 2)]
    
    # Se llama a la función experiments_knn para realizar la validación cruzada
    for use_paralellization in [False, True]:
        if use_paralellization:
            # Experimentamos con diferentes valores de n_jobs
            for n_jobs in range(2, 12 + 1):
                start = time.time()
                results_with_p_pd = experiments_knn(ks, X_train_scaled, y_train, 10, parallelism=True, n_jobs=n_jobs)
                end = time.time()
                print(f"Tiempo de ejecución con paralelización: {n_jobs} jobs", end - start)
                print(f"Los dataframes resultantes {"no " if not(results_without_p_pd.equals(results_with_p_pd)) else ""}son iguales")
        else:
            start = time.time()
            results_without_p_pd = experiments_knn(ks, X_train_scaled, y_train, 10)
            end = time.time()
            print(f"Tiempo de ejecución sin paralelización: ", end - start)

