import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

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
    
def seleccion_de_variables(X_encoded: pd.DataFrame, y: pd.Series, n: int) -> list[str]:
    """
    Calcula a correlación de cada variable co target e devolve as n con maior
    correlación absoluta.
 
    Parámetros
    ----------
    X_encoded : DataFrame con todas as variables (xa codificadas).
    y         : Serie co target.
    n         : Número de variables a seleccionar.
 
    Devolve
    -------
    Lista de n nomes de columnas ordenadas de maior a menor correlación.
    """
    df_corr = X_encoded.copy()
    df_corr['_target_'] = y.values
 
    correlacions = df_corr.corr()['_target_'].abs().sort_values(ascending=False)
 
    # O índice 0 é a correlación do target consigo mesmo (1.0), saltámolo
    features = correlacions.index[1:n + 1].tolist()
 
    print(f"--- TOP {n} Variables ---")
    for feat in features:
        print(f"  · {feat}  (corr: {correlacions[feat]:.4f})")
    print("-" * 35 + "\n")
 
    return features

def preprocesar_datos(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Preprocesa os datos de adestramento e test realizando os seguintes pasos:
    1. Separa a variable obxectivo (Target_Risco).
    2. Elimina as columnas non útiles ou problemáticas (ID_Cliente, Data_Solicitude).
    3. Aplica One-Hot Encoding ás variables categóricas de texto.
    4. Aliña os datasets para garantir que test ten as mesmas columnas ca train.
    5. Imputa os valores nulos coa mediana do adestramento.
    
    Parámetros
    ----------
    train : DataFrame cos datos de adestramento orixinais.
    test  : DataFrame cos datos de test orixinais.
    
    Devolve
    -------
    X_train_aligned : DataFrame de adestramento preprocesado.
    X_test_aligned  : DataFrame de test preprocesado.
    y_train         : Serie coa variable obxectivo.
    """
    print("Iniciando preprocesado...")
    
    # 1. Separar o target
    y_train = train['Target_Risco'].copy()
    
    # 2. Eliminar columnas que non aportan ou xeran ruído
    cols_a_borrar_train = ['ID_Cliente', 'Target_Risco', 'Data_Solicitude']
    cols_a_borrar_test = ['ID_Cliente', 'Data_Solicitude']
    
    X_train_raw = train.drop(columns=[col for col in cols_a_borrar_train if col in train.columns], errors='ignore')
    X_test_raw = test.drop(columns=[col for col in cols_a_borrar_test if col in test.columns], errors='ignore')
    
    # 3. One-Hot Encoding
    X_train_encoded = pd.get_dummies(X_train_raw)
    X_test_encoded = pd.get_dummies(X_test_raw)
    
    # 4. Aliñamento para evitar diferenzas nas variables categóricas
    X_train_aligned, X_test_aligned = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)
    
    # 5. Imputación de nulos coa mediana
    for col in X_train_aligned.columns:
        mediana = X_train_aligned[col].median()
        X_train_aligned[col] = X_train_aligned[col].fillna(mediana)
        X_test_aligned[col]  = X_test_aligned[col].fillna(mediana)
        
    print(f"Preprocesado rematado. Variables finais: {X_train_aligned.shape[1]}")
    return X_train_aligned, X_test_aligned, y_train

if __name__ == "__main__":
    df_train = pd.read_csv("data/train.csv")
    print(df_train.head())
    print(df_train.shape)
    representar_boxplots(df_train, 'Target_Risco', 'Boxplots das variables numéricas e histograma do risco')