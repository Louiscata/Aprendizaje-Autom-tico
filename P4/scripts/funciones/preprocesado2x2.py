import pandas as pd
import numpy as np

# ── Constantes Globais de Preprocesado ───────────────────────────────────────

# Variables lixo que comprobamos que só meten ruído
MALARDAS = ['ID_Cliente', 'Lonxitude_Nome', 'Tempo_Web_Minutos', 'Subscricion_Email']

# Variables categóricas que non teñen sentido matemático (falso continuo)
CATEGORICAS = ['Profesion', 'Tipo_Dispositivo', 'Dia_Solicitude', 'Codigo_Postal', 'Mes_Solicitude']

def crear_features(df_input):
    df = df_input.copy()


    # Asegurarnos de que a data é tipo datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Data_Solicitude']):
        df['Data_Solicitude'] = pd.to_datetime(df['Data_Solicitude'])

    df['Mes_Solicitude'] = df['Data_Solicitude'].dt.month


    eps = 0.1
    
    df['ratio_debeda_ingresos'] = (
        df['Debeda_Total'] / (df['Ingresos_Anuais'] + eps)
    )
    df['patrimonio_neto'] = df['Patrimonio_Total'] - df['Debeda_Total']

    ingreso_mensual = df['Ingresos_Anuais'] / 12 + eps
    df['ratio_saldo_mensual'] = df['Saldo_Medio_3M'] / ingreso_mensual

    df['ratio_limite_ingresos'] = (
        df['Limite_Credito_Total'] / (df['Ingresos_Anuais'] + eps)
    )

    patrimonio_neto_pos = df['patrimonio_neto'].clip(lower=eps)
    df['ratio_cota_patrimonio'] = (
        df['Cota_Mensual_Prestamos'] / patrimonio_neto_pos
    )
    
    return df


def preprocesar_datos(train_raw, test_raw, usar_ohe=False):
    """
    Motor de preprocesado definitivo.
    1. Limpa duplicados e crea variables financeiras.
    2. Elimina as variables lixo.
    3. Imputa nulos (Mediana para numéricas, 'Descoñecido' para texto).
    4. Aplica OHE se se solicita.
    """
    
    # 1. Copias de seguridade e limpeza de duplicados no train
    train = train_raw.copy()
    test = test_raw.copy()
    
    n_antes = len(train)
    train = train.drop_duplicates(keep='first').reset_index(drop=True)
    if (n_antes - len(train)) > 0:
        print(f"  -> Eliminados {n_antes - len(train)} duplicados.")

    # 2. Enxeñaría de variables (Ratios financeiras)
    train = crear_features(train)
    test  = crear_features(test)
    
    # Extraemos a variable obxectivo
    y_train = train['Target_Risco'].copy()
    
    # 3. Borramos variables inútiles e as que xa non fan falta no adestramento
    cols_a_borrar = MALARDAS + ['Target_Risco', 'Data_Solicitude']
    X_train = train.drop(columns=[c for c in cols_a_borrar if c in train.columns])
    X_test  = test.drop(columns=[c for c in cols_a_borrar if c in test.columns])
    
    # 4. Preparación das categóricas e imputación de Nulos
    for col in CATEGORICAS:
        if col in X_train.columns:
            # Protelemos o Código Postal para que non pareza un número
            if col == 'Codigo_Postal':
                X_train[col] = 'CP_' + X_train[col].astype(str)
                X_test[col]  = 'CP_' + X_test[col].astype(str)
            
            # Enchemos nulos nas categóricas
            X_train[col] = X_train[col].fillna("DESCONECIDO").astype(str)
            X_test[col]  = X_test[col].fillna("DESCONECIDO").astype(str)

    # Imputar Nulos para as numéricas coa mediana
    cols_numericas = [c for c in X_train.columns if c not in CATEGORICAS]
    for col in cols_numericas:
        if X_train[col].dtype in [np.float64, np.int64]:
            mediana = X_train[col].median()
            X_train[col] = X_train[col].fillna(mediana)
            X_test[col]  = X_test[col].fillna(mediana)

    # 5. Xestión do One-Hot Encoding
    if usar_ohe:
        print(f"  -> Aplicando One-Hot Encoding ás variables: {CATEGORICAS}")
        X_train = pd.get_dummies(X_train, columns=CATEGORICAS, dtype=int)
        X_test  = pd.get_dummies(X_test, columns=CATEGORICAS, dtype=int)
        
        # Aliñamos as columnas por se algunha categoría só existe no train e non no test
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    else:
        print("  -> OHE desactivado. Preparado para motor CatBoost nativo.")
        
    print(f"  -> Variables finais xeradas: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, CATEGORICAS