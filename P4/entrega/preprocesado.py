import pandas as pd
import numpy as np

# Columnas que demostraron non aportar nada útil ao modelo
MALARDAS = ['ID_Cliente', 'Lonxitude_Nome', 'Tempo_Web_Minutos', 'Subscricion_Email']

# CatBoost pode traballar con estas directamente sen codificalas
CATEGORICAS = ['Profesion', 'Tipo_Dispositivo', 'Dia_Solicitude', 'Codigo_Postal', 'Mes_Solicitude']

def crear_features(df_input):
    df = df_input.copy()

    # Aseguramos que a data está en formato correcto antes de extraer o mes
    if not pd.api.types.is_datetime64_any_dtype(df['Data_Solicitude']):
        df['Data_Solicitude'] = pd.to_datetime(df['Data_Solicitude'])

    # Mes como variable categórica, pode capturar estacionalidade
    df['Mes_Solicitude'] = df['Data_Solicitude'].dt.month

    # eps pequeno para evitar divisións por cero en clientes con ingresos nulos
    eps = 0.1

    # Canto debe o cliente en proporción aos seus ingresos
    df['ratio_debeda_ingresos'] = df['Debeda_Total'] / (df['Ingresos_Anuais'] + eps)
    # Riqueza real: o que ten menos o que debe
    df['patrimonio_neto'] = df['Patrimonio_Total'] - df['Debeda_Total']

    ingreso_mensual = df['Ingresos_Anuais'] / 12 + eps
    # Como de folgado está o cliente a fin de mes
    df['ratio_saldo_mensual'] = df['Saldo_Medio_3M'] / ingreso_mensual
    # Canto crédito ten dispoñible respecto ao que gaña
    df['ratio_limite_ingresos'] = df['Limite_Credito_Total'] / (df['Ingresos_Anuais'] + eps)

    # clip para evitar patrimonio negativo no denominador
    patrimonio_neto_pos = df['patrimonio_neto'].clip(lower=eps)
    # Que porcentaxe do seu patrimonio vai en cotas de préstamos
    df['ratio_cota_patrimonio'] = df['Cota_Mensual_Prestamos'] / patrimonio_neto_pos

    return df


def preprocesar_datos(train_raw, test_raw, usar_ohe=False):
    train = train_raw.copy()
    test = test_raw.copy()

    # Eliminamos filas repetidas, só en train (o test non se toca)
    n_antes = len(train)
    train = train.drop_duplicates(keep='first').reset_index(drop=True)
    if (n_antes - len(train)) > 0:
        print(f"  -> Eliminados {n_antes - len(train)} duplicados.")

    train = crear_features(train)
    test  = crear_features(test)

    y_train = train['Target_Risco'].copy()

    # Borramos o lixo e as columnas auxiliares que xa non fan falta
    cols_a_borrar = MALARDAS + ['Target_Risco', 'Data_Solicitude']
    X_train = train.drop(columns=[c for c in cols_a_borrar if c in train.columns])
    X_test  = test.drop(columns=[c for c in cols_a_borrar if c in test.columns])

    for col in CATEGORICAS:
        if col in X_train.columns:
            # Sen este prefixo CatBoost trataría o CP como un número continuo
            if col == 'Codigo_Postal':
                X_train[col] = 'CP_' + X_train[col].astype(str)
                X_test[col]  = 'CP_' + X_test[col].astype(str)

            # Nulos categóricos    categoría explícita en vez de NaN
            X_train[col] = X_train[col].fillna("DESCONECIDO").astype(str)
            X_test[col]  = X_test[col].fillna("DESCONECIDO").astype(str)

    # Nulos numéricos    mediana calculada sempre sobre train, aplicada en ambos
    cols_numericas = [c for c in X_train.columns if c not in CATEGORICAS]
    for col in cols_numericas:
        if X_train[col].dtype in [np.float64, np.int64]:
            mediana = X_train[col].median()
            X_train[col] = X_train[col].fillna(mediana)
            X_test[col]  = X_test[col].fillna(mediana)

    if usar_ohe:
        print(f"  -> Aplicando One-Hot Encoding ás variables: {CATEGORICAS}")
        X_train = pd.get_dummies(X_train, columns=CATEGORICAS, dtype=int)
        X_test  = pd.get_dummies(X_test, columns=CATEGORICAS, dtype=int)

        # Se algunha categoría non aparece no test, aliñamos con ceros
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    else:
        print("  -> OHE desactivado. Preparado para motor CatBoost nativo.")

    print(f"  -> Variables finais xeradas: {X_train.shape[1]}")

    return X_train, X_test, y_train, CATEGORICAS