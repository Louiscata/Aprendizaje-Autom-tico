import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. Restriccións de dominio e Listas de Variables
# ─────────────────────────────────────────────────────────────────────────────

_RESTRICCIONS: dict = {
    'ID_Cliente':                  (100_000, None),
    'Data_Solicitude':             ('2021-01-01', '2023-12-31'),
    'Idade':                       (18, 85),
    'Lonxitude_Nome':              (10, 35),
    'Num_Fillos':                  (0, 12),
    'Profesion':                   {'Asalariado', 'Autónomo', 'Funcionario',
                                    'Estudante', 'Desempregado'},
    'Anos_Emprego':                (0.0, 62.0),
    'Ingresos_Anuais':             (0, None),
    'Tipo_Dispositivo':            {'Windows', 'MacOS', 'iOS', 'Android', 'Linux'},
    'Tempo_Web_Minutos':           (1, 50),
    'Subscricion_Email':           {0, 1},
    'Dia_Solicitude':              {'Luns', 'Martes', 'Mércores', 'Xoves',
                                    'Venres', 'Sábado', 'Domingo'},
    'Distancia_Oficina_Km':        (0, None),
    'Codigo_Postal':               (0, None),
    'Patrimonio_Total':            (0, None),
    'Debeda_Total':                (0, None),
    'Historial_Impagos':           {0, 1},
    'Numero_Tarxetas':             (0, 12),
    'Utilizacion_Credito':         (0.0, 1.2),
    'Consultas_Risco_6M':          (0, 17),
    'Limite_Credito_Total':        (0, None),
    'Cota_Mensual_Prestamos':      (0, None),
    'Ratio_Cota_Ingresos':         (0.0, 2.5),
    'Prestamos_Activos':           (0, 9),
    'Antiguedade_Cliente_Anos':    (0.0, 40.0),
    'Saldo_Medio_3M':              (0, None),
    'Variacion_Saldo_6M':          (-1.0, 1.0),
    'Fondo_Emerxencia_Meses':      (0.0, 24.0),
    'Indice_Estres_Financeiro':    (0.0, 2.0),
    'Target_Risco':                {0, 1, 2, 3},
}

_COLS_OUTLIERS = [
    'Idade', 'Lonxitude_Nome', 'Anos_Emprego', 'Ingresos_Anuais',
    'Tempo_Web_Minutos', 'Distancia_Oficina_Km', 'Patrimonio_Total',
    'Debeda_Total', 'Utilizacion_Credito', 'Limite_Credito_Total',
    'Cota_Mensual_Prestamos', 'Ratio_Cota_Ingresos', 'Antiguedade_Cliente_Anos',
    'Saldo_Medio_3M', 'Variacion_Saldo_6M', 'Fondo_Emerxencia_Meses',
    'Indice_Estres_Financeiro',
    # Variables derivadas
    'ratio_debeda_ingresos', 'patrimonio_neto', 'ratio_saldo_mensual', 
    'ratio_limite_ingresos', 'ratio_cota_patrimonio'
]

_COLS_TRANSFORMACION_LOG = [
    'Ingresos_Anuais', 'Patrimonio_Total', 'Debeda_Total',
    'Limite_Credito_Total', 'Saldo_Medio_3M', 'Cota_Mensual_Prestamos',
    # Variables derivadas
    'ratio_debeda_ingresos', 'ratio_saldo_mensual',
    'ratio_limite_ingresos', 'ratio_cota_patrimonio'
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Selección de Variables (Volve correlacion en vez de mutual info)
# ─────────────────────────────────────────────────────────────────────────────

def seleccion_de_variables(X_encoded: pd.DataFrame, y: pd.Series, n: int) -> list[str]:
    df_corr = X_encoded.copy()
    df_corr['_target_'] = y.values
 
    correlacions = df_corr.corr()['_target_'].abs().sort_values(ascending=False)
    features = correlacions.index[1:n + 1].tolist()
 
    print(f"--- Mellores {n} variables por correlación ---")
    for feat in features:
        print(f"  - {feat}  (corr: {correlacions[feat]:.4f})")
    print("-" * 40 + "\n")

    return features

# ─────────────────────────────────────────────────────────────────────────────
# 3. Enxeñaría de Variables
# ─────────────────────────────────────────────────────────────────────────────

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engade novas variables derivadas (Ratios e Temporais)."""
    df = df.copy()
    eps = 1.0 
    
    df['ratio_debeda_ingresos'] = df['Debeda_Total'] / (df['Ingresos_Anuais'] + eps)
    df['patrimonio_neto'] = df['Patrimonio_Total'] - df['Debeda_Total']
    
    ingreso_mensual = df['Ingresos_Anuais'] / 12 + eps
    df['ratio_saldo_mensual'] = df['Saldo_Medio_3M'] / ingreso_mensual
    df['ratio_limite_ingresos'] = df['Limite_Credito_Total'] / (df['Ingresos_Anuais'] + eps)
    
    patrimonio_neto_pos = df['patrimonio_neto'].clip(lower=eps)
    df['ratio_cota_patrimonio'] = df['Cota_Mensual_Prestamos'] / patrimonio_neto_pos

    if 'Data_Solicitude' in df.columns:
        datas = pd.to_datetime(df['Data_Solicitude'], errors='coerce')
        df['mes_solicitude']       = datas.dt.month
        df['trimestre_solicitude'] = datas.dt.quarter

    n_novos = 5 + (2 if 'Data_Solicitude' in df.columns else 0)
    print(f" {n_novos} novas variables derivadas engadidas.\n")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 4. Preprocesado principal
# ─────────────────────────────────────────────────────────────────────────────

def preprocesar_datos(
    train: pd.DataFrame,
    test: pd.DataFrame,
    umbral_nan: int = 10,
    outliers: str = None, 
    imputar_nan_test: bool = False,
    normalizar: bool = True,
    trans_log: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    
    print("=" * 60)
    print("  PREPROCESADO DE DATOS")
    print("=" * 60)

    train = train.copy()
    test  = test.copy()

    # Limpeza
    train = _eliminar_duplicados(train)
    train = _eliminar_imposibles(train)

    if outliers == 'eliminar':
        train, _ = _eliminar_outliers_iqr(train)
    elif outliers == 'capear':
        train, test = _capear_outliers_iqr(train, test)
    else:
        print("Detección de outliers desactivada.\n")

    train, medianas = _xestionar_nans_train(train, umbral_nan)

    # Imputar Test
    if imputar_nan_test:
        print("[Test] Imputando NaN coas medianas de train...")
        for col in test.columns:
            if test[col].isna().any() and col in medianas.index:
                test[col] = test[col].fillna(medianas[col])

    # Feature Engineering
    print("Creando variables derivadas...")
    train = crear_features(train)
    test  = crear_features(test)

    y_train = train['Target_Risco'].copy()

    cols_drop_train = ['ID_Cliente', 'Data_Solicitude', 'Target_Risco']
    cols_drop_test  = ['ID_Cliente', 'Data_Solicitude']

    X_train_raw = train.drop(columns=[c for c in cols_drop_train if c in train.columns])
    X_test_raw  = test.drop(columns=[c for c in cols_drop_test  if c in test.columns])

    # Para que funcione CP con OHE
    for df_ in [X_train_raw, X_test_raw]:
        if 'Codigo_Postal' in df_.columns:
            df_['Codigo_Postal'] = 'CP_' + df_['Codigo_Postal'].astype(int).astype(str)

    # OHE e Aliñamento
    X_train_encoded = pd.get_dummies(X_train_raw)
    X_test_encoded  = pd.get_dummies(X_test_raw)
    X_train_aligned, X_test_aligned = X_train_encoded.align(
        X_test_encoded, join='left', axis=1, fill_value=0
    )

    # Transformacións Opcionais
    if trans_log:
        X_train_aligned, _ = _aplicar_transformacion_log(X_train_aligned, _COLS_TRANSFORMACION_LOG)
        X_test_aligned, _  = _aplicar_transformacion_log(X_test_aligned, _COLS_TRANSFORMACION_LOG)

    if normalizar:
        X_train_aligned, X_test_aligned = _normalizar(X_train_aligned, X_test_aligned)
    else:
        print("Normalización desactivada.\n")

    print(f"  Train final : {X_train_aligned.shape[0]} filas - {X_train_aligned.shape[1]} variables")
    print(f"  Test final  : {X_test_aligned.shape[0]} filas - {X_test_aligned.shape[1]} variables")
    print("=" * 60 + "\n")

    return X_train_aligned, X_test_aligned, y_train

# ─────────────────────────────────────────────────────────────────────────────
# 5. Funcións auxiliares (Privadas)
# ─────────────────────────────────────────────────────────────────────────────

def _eliminar_duplicados(train: pd.DataFrame) -> pd.DataFrame:
    print("Eliminando duplicados en train")
    n_antes = len(train)
    train = train.drop_duplicates(keep='first').reset_index(drop=True)
    if (n_antes - len(train)) > 0:
        print(f" - {n_antes - len(train)} filas duplicadas eliminadas\n")
    return train

def _eliminar_imposibles(train: pd.DataFrame) -> pd.DataFrame:
    print("Eliminando valores imposibles")
    for col, restricion in _RESTRICCIONS.items():
        if col not in train.columns: continue
        if col == 'Data_Solicitude':
            mascara = _mascara_data_imposible(train[col], restricion)
        elif isinstance(restricion, set):
            mascara = _mascara_categorica_imposible(train[col], restricion)
        else:
            mascara = _mascara_numerica_imposible(train[col], restricion)

        if mascara.sum() > 0:
            train.loc[mascara, col] = np.nan
            print(f" - {mascara.sum()} valores imposibles en '{col}' convertidos a NaN")
        else:
            print(f" - Sen valores imposibles detectados en {col}")
    return train

def _eliminar_outliers_iqr(train: pd.DataFrame, factor: float=1.5):
    print(f"Eliminando outliers con IQR * {factor})")
    limites = {}
    for col in _COLS_OUTLIERS:
        if col not in train.columns: continue
        q1, q3 = train[col].quantile(0.25), train[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        limites[col] = (lo, hi)
        train.loc[(train[col] < lo) | (train[col] > hi), col] = np.nan
    return train, limites

def _capear_outliers_iqr(train: pd.DataFrame, test: pd.DataFrame, factor: float=3.0):
    print(f"Capeando outliers con IQR * {factor}")
    for col in _COLS_OUTLIERS:
        if col not in train.columns: continue
        q1, q3 = train[col].quantile(0.25), train[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        train[col] = train[col].clip(lower=lo, upper=hi)
        if col in test.columns:
            test[col] = test[col].clip(lower=lo, upper=hi)
    return train, test

def _xestionar_nans_train(train: pd.DataFrame, umbral: int):
    print(f"Xestionando NaN")
    mascara = train.isna().sum(axis=1) > umbral
    train.drop(index=train.index[mascara], inplace=True)
    medianas = train.median(numeric_only=True)
    for col in train.columns:
        if train[col].isna().any():
            train[col] = train[col].fillna(medianas.get(col, 0))
    return train, medianas

def _normalizar(X_train: pd.DataFrame, X_test: pd.DataFrame):
    print("Normalizando variables a N(0,1)")
    cols_onehot = [col for col in X_train.columns if set(X_train[col].dropna().unique()).issubset({0, 1})]
    for col in X_train.columns:
        if col not in cols_onehot:
            std = X_train[col].std()
            if std > 0:
                media = X_train[col].mean()
                X_train[col] = (X_train[col] - media) / std
                X_test[col]  = (X_test[col] - media) / std
    return X_train, X_test

def _aplicar_transformacion_log(df: pd.DataFrame, columnas_log: list):
    print("Aplicando Log1p...")
    for col in columnas_log:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df, len(columnas_log)

def _mascara_numerica_imposible(serie: pd.Series, restricion: tuple):
    lo, hi = restricion
    mascara = pd.Series(False, index=serie.index)
    if lo is not None: mascara |= serie.notna() & (serie < lo)
    if hi is not None: mascara |= serie.notna() & (serie > hi)
    return mascara

def _mascara_categorica_imposible(serie: pd.Series, valores_validos: set):
    return serie.notna() & ~serie.isin(valores_validos)

def _mascara_data_imposible(serie: pd.Series, restricion: tuple):
    datas = pd.to_datetime(serie, errors='coerce')
    return datas.notna() & ((datas < pd.Timestamp(restricion[0])) | (datas > pd.Timestamp(restricion[1])))