# Importaciones
# Clasificador a usar: KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

# Método de validación cruzada: KFold
from sklearn.model_selection import KFold

# Librerías para manipular datasets
import numpy as np
import pandas as pd

# Librería para calcular el accuracy
from sklearn.metrics import accuracy_score

def experiments_kFold(model, X, y, n_splits):
    """
    Esta función realiza la validación cruzada de un modelo (De forma secuencial) y retorna la media y la desviación estándar de los resultados
    model: Modelo preconfigurado (Ya se le pasaron los hiperparámetros)
    X: Matriz de características (Ya deben estar escalados)
    y: Vector de clases
    n_splits: Número de particiones para la validación cruzada
    return: Media y desviación estándar de los resultados
    """
    accuracies = []

    # Se crea el método de validación cruzada
    kfold = KFold(n_splits=n_splits)

    # Se realiza la validación cruzada
    for train_index, test_index in kfold.split(X):
        # Se calcula el accuracy
        accuracy = apply_one_fold(model, X, y, train_index, test_index)

        # Se agrega el accuracy a la lista
        accuracies.append(accuracy)

    # Se calcula la media y la desviación estándar de los resultados y se retornan
    return str(model.get_params()), np.mean(accuracies), np.std(accuracies)

def apply_one_fold(model, X, y, train_index, test_index):
    """
    Esta función entrena y evalúa un modelo con un fold específico
    model: Modelo preconfigurado (Ya se le pasaron los hiperparámetros)
    X: Matriz de características (Ya deben estar escalados)
    y: Vector de clases
    train_index: Índices de entrenamiento
    test_index: Índices de prueba
    return: Accuracy
    """
    # Se separan los datos de entrenamiento y prueba
    X_train_kfold, X_test_kfold = X[train_index], X[test_index]
    y_train_kfold, y_test_kfold = y[train_index], y[test_index]

    # Se entrena el modelo
    model.fit(X_train_kfold, y_train_kfold)

    # Se evalúa el modelo
    y_pred = model.predict(X_test_kfold)

    # Se calcula el accuracy
    accuracy = accuracy_score(y_test_kfold, y_pred)

    # Se retorna el accuracy
    return accuracy 

def experiments_knn(ks, X, y, n_splits, parallelism=False, n_jobs=4):
    """
    Esta función preconfigura el modelo KNN con diferentes valores de k y llama a la función experiments para realizar la validación cruzada
    ks: Lista de valores de k
    X: Matriz de características (Ya deben estar escalados)
    y: Vector de clases
    n_splits: Número de particiones para la validación cruzada
    """
    # Se crea la lista para almacenar los resultados
    results = dict()

    if parallelism:
        # Se llama a la función experiments para realizar la validación cruzada
        # En el enfoque paralelizado, se le pide al usuario que pase los modelos ya preconfigurados
        results = experiments_parallelism([KNeighborsClassifier(n_neighbors=k) for k in ks], X, y, n_splits, n_jobs)
    else:
        # Se recorren los valores de k
        for k in ks:
            # Se crea el modelo
            model = KNeighborsClassifier(n_neighbors=k)

            # Se llama a la función experiments para realizar la validación cruzada
            _, mean, std = experiments_kFold(model, X, y, n_splits)

            # Se agrega el resultado al diccionario
            results[str(model.get_params())] = (mean, std)

    # Se retorna la lista de resultados
    df = pd.DataFrame(results)

    if not parallelism:
        # Se transpone el dataframe
        df = df.T

    else:
        df = df.set_index(0)

    # Se renombran las columnas
    df.columns = ["Media", "Desviación estándar"]
    return df

def experiments_parallelism(models, X, y, n_splits, n_jobs=4):
    """
    Esta función preconfigura el modelo KNN con diferentes valores de k y llama a la función experiments para realizar la validación cruzada
    ks: Lista de valores de k
    X: Matriz de características (Ya deben estar escalados)
    y: Vector de clases
    n_splits: Número de particiones para la validación cruzada
    """
    # Se importa la librería para realizar el paralelismo
    import multiprocessing as mp

    # Se crea el pool de procesos
    pool = mp.Pool(n_jobs)

    # Se llama a la función experiments para realizar la validación cruzada
    results = pool.starmap(experiments_kFold, [(model, X, y, n_splits) for model in models])

    # Se cierra el pool de procesos
    pool.close()
    # Se retorna la lista de resultados
    return results
