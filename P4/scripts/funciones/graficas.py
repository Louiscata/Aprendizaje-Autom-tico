import pandas as pd
import matplotlib.pyplot as plt
import math

def representar_histogramas(df, nome_etiquetas, titulo):
    """
    Debuxa un grid de histogramas para todas as variables numéricas dun DataFrame,
    e remata cunha gráfica de barras para a variable obxectivo.
    """
    # Filtramos para quedarnos unicamente cas variables numéricas
    df_numerico = df.select_dtypes(include=['number'])
    
    # Seleccionamos as variables preditoras numéricas, quitando o Target e o ID
    columnas_variables = df_numerico.drop(columns=[nome_etiquetas, 'ID_Cliente'], errors='ignore').columns
    n_variables = len(columnas_variables)

    # O total de gráficas será o número de variables + 1 (a gráfica de barras final)
    total_graficas = n_variables + 1

    # Calculamos as filas necesarias
    columnas_grid = 5
    filas_grid = math.ceil(total_graficas / columnas_grid)

    # Creamos a figura co tamaño adaptado ao número de filas
    fig, axes = plt.subplots(nrows=filas_grid, ncols=columnas_grid, figsize=(15, 4 * filas_grid))
    fig.suptitle(titulo, fontsize=16, y=1.02)

    # Aplanamos os eixes para iterar facilmente
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    # 1. Debuxamos os histogramas iterando por cada recadro
    for i, col in enumerate(columnas_variables):
        # Usamos 20 'bins' (barras) por defecto, podes cambialo se precisas máis detalle
        df_numerico[col].plot(kind='hist', ax=axes[i], bins=20, color='skyblue', edgecolor='black')
        axes[i].set_title(col)
        axes[i].set_ylabel('Frecuencia')
        axes[i].set_xlabel('Valor')

    # 2. Debuxamos a gráfica de barras xusto no recadro seguinte para o Target
    ax_bar = axes[n_variables]

    # Comprobamos se a etiqueta existe no df (por se lle pasas o dataset de test sen querer)
    if nome_etiquetas in df.columns:
        conteos = df[nome_etiquetas].value_counts().sort_index()
        cores_risco = ['green', 'gold', 'darkorange', 'red']
        cores = cores_risco[:len(conteos)]

        ax_bar.bar(conteos.index, conteos.values, color=cores, edgecolor='black')
        ax_bar.set_title(f'Distribución de {nome_etiquetas}')
        ax_bar.set_xticks(conteos.index)
        ax_bar.set_xlabel(nome_etiquetas)
        ax_bar.set_ylabel('Frecuencia')
    else:
        ax_bar.set_visible(False) # Ocultamos se non hai target

    # 3. Ocultar recadros baleiros sobrantes na cuadrícula
    for j in range(total_graficas, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

def representar_boxplots(df, nome_etiquetas, titulo):
    # Filtramos para quedarnos unicamente cas variables numéricas
    df_numerico = df.select_dtypes(include=['number'])
    
    # Seleccionamos as variables preditoras numéricas, quitando o Target e o ID
    columnas_variables = df_numerico.drop(columns=[nome_etiquetas, 'ID_Cliente'], errors='ignore').columns
    n_variables = len(columnas_variables)

    # O total de gráficas será o número de variables + 1 (a gráfica de barras final)
    total_graficas = n_variables + 1

    # Calculamos as filas necesarias
    columnas_grid = 5
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

if __name__ == "__main__":
    df_train = pd.read_csv("data/train.csv")
    print(df_train.head())
    print(df_train.shape)
    representar_histogramas(df_train, 'Target_Risco', 'Histogramas')