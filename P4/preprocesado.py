import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

df_train = pd.read_csv("data/train.csv")

print(df_train.head())
print(df_train.shape)

def representar_boxplots(df, nome_etiquetas, titulo):
    # Filtramos para quedarnos unicamente cas variables numéricas
    df_numerico = df.select_dtypes(include=['number'])
    
    # Seleccionamos as variables preditoras numéricas, quitando o Target e o ID
    columnas_variables = df_numerico.drop(columns=[nome_etiquetas, 'ID_Cliente'], errors='ignore').columns
    n_variables = len(columnas_variables)

    # O total de gráficas será o número de variables + 1 (a gráfica de barras final)
    total_graficas = n_variables + 1

    # Calculamos as filas necesarias
    columnas_grid = 3
    filas_grid = math.ceil(total_graficas / columnas_grid)

    # Creamos a figura co tamaño adaptado ao número de filas
    fig, axes = plt.subplots(nrows=filas_grid, ncols=columnas_grid, figsize=(15, 4 * filas_grid))
    fig.suptitle(titulo, fontsize=16, y=1.02)

    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    # 1. Debuxamos os boxplots iterando por cada recadro
    for i, col in enumerate(columnas_variables):
        df.boxplot(column=col, ax=axes[i], grid=False)
        axes[i].set_title(col)

    # 2. Debuxamos a gráfica de barras xusto no recadro seguinte
    ax_bar = axes[n_variables]

    conteos = df[nome_etiquetas].value_counts().sort_index()
    
    # Asignamos cores en base ás clases de risco (de 0 a 3)
    cores_risco = ['green', 'gold', 'darkorange', 'red']
    cores = cores_risco[:len(conteos)]

    ax_bar.bar(conteos.index, conteos.values, color=cores, edgecolor='black')
    ax_bar.set_title('Histograma de obxectivo')
    ax_bar.set_xticks(conteos.index)
    ax_bar.set_xlabel(nome_etiquetas)

    # 3. Ocultar recadros baleiros sobrantes
    for j in range(total_graficas, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()