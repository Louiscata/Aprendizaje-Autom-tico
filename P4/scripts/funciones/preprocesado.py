import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

# ─────────────────────────────────────────────────────────────────────────────
# Restriccións de dominio (sen cambios)
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
    'Tipo_Dispositivo':            {'PC_Windows', 'Mac', 'iPhone', 'Android', 'Linux'},
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

# Variables que ao velas no histograma teñen unha gran acumulación próxima ao cero e unha longa cola cara a dereita, o que suxire que unha transformación logarítmica pode mellorar a súa distribución e a súa relación co obxectivo.

_COLS_TRANSFORMACION_LOG = [
    'Ingresos_Anuais',
    'Patrimonio_Total',
    'Debeda_Total',
    'Limite_Credito_Total',
    'Saldo_Medio_3M',
    'Cota_Mensual_Prestamos',
    # Variables derivadas
    'ratio_debeda_ingresos',
    'ratio_saldo_mensual',
    'ratio_limite_ingresos',
    'ratio_cota_patrimonio'
]


# ─────────────────────────────────────────────────────────────────────────────
# Función de selección de variables  [FIX] Pearson -> Mutual Information
# ─────────────────────────────────────────────────────────────────────────────

def seleccion_de_variables(
    X_encoded: pd.DataFrame,
    y: pd.Series,
    n: int,
    random_state: int = 42,
) -> list[str]:
    """
    Selecciona as n variables con maior información mutua co target.

    A información mutua (MI) mide dependencia non-lineal e é apropiada para
    targets de clasificación multiclase, a diferenza da correlación de Pearson
    que só captura relacións lineais e trata o target como continuo.

    Parámetros
    ----------
    X_encoded    : DataFrame con todas as variables (xa codificadas/numéricas).
    y            : Serie co target (enteiros 0..K-1).
    n            : Número de variables a seleccionar.
    random_state : Semente para reproducibilidade do estimador MI.

    Devolve
    -------
    Lista de n nomes de columnas ordenadas de maior a menor MI.
    """
    # mutual_info_classif require arrays numpy sen NaN
    X_arr = X_encoded.values.astype(float)
    mi_scores = mutual_info_classif(X_arr, y, random_state=random_state)
    mi_series = pd.Series(mi_scores, index=X_encoded.columns).sort_values(ascending=False)

    features = mi_series.index[:n].tolist()

    print(f"--- TOP {n} Variables (Mutual Information) ---")
    for feat in features:
        print(f"  - {feat}  (MI: {mi_series[feat]:.4f})")
    print("-" * 45 + "\n")

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Función de inxeniería de variables  [NOVO]
# ─────────────────────────────────────────────────────────────────────────────

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engade novas variables derivadas ao DataFrame.

    As transformacións aplícanse sobre o DataFrame *antes* de eliminar columnas
    nin de codificar categóricas, polo que deben chamarse ao comezo do pipeline
    (logo da limpeza, antes do encoding).

    Features creados
    ----------------
    Ratios financeiros:
      - ratio_debeda_ingresos   : Debeda total relativa aos ingresos anuais.
                                  É o 2º feature con maior MI no dataset.
      - patrimonio_neto         : Patrimonio Total menos Débeda Total.
      - ratio_saldo_mensual     : Saldo medio mensual vs ingreso mensual.
      - ratio_limite_ingresos   : Límite de crédito relativo aos ingresos.
      - ratio_cota_patrimonio   : Cota mensual relativa ao patrimonio neto.

    Temporais (extrae de Data_Solicitude):
      - mes_solicitude          : Mes (1–12), pode ter estacionalidade.
      - trimestre_solicitude    : Trimestre (1–4).
      - ano_solicitude          : Ano (2021, 2022, 2023).
    """
    df = df.copy()
    eps = 1.0   # evita divisións por cero

    # ── Ratios financeiros ────────────────────────────────────────────────────
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

    # ── Temporais ─────────────────────────────────────────────────────────────
    if 'Data_Solicitude' in df.columns:
        datas = pd.to_datetime(df['Data_Solicitude'], errors='coerce')
        df['mes_solicitude']       = datas.dt.month
        df['trimestre_solicitude'] = datas.dt.quarter
        df['ano_solicitude']       = datas.dt.year

    n_novos = 5 + (3 if 'Data_Solicitude' in df.columns else 0)
    print(f"[Feature Eng.] {n_novos} novas variables engadidas.\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Función principal de preprocesado
# ─────────────────────────────────────────────────────────────────────────────

def preprocesar_datos(
    train: pd.DataFrame,
    test: pd.DataFrame,
    umbral_nan: int = 10,
    outliers: str = None, # 'eliminar' ou 'capear'
    imputar_nan_test: bool = True,
    normalizar: bool = True,
    trans_log: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Preprocesa os datos de adestramento e test.

    O test NON se limpa: non se eliminan duplicados, valores imposibles,
    outliers nin filas con moitos NaN. O test só recibe:
      - Inxeniería de variables (mesmas operacións que en train).
      - Imputación de NaN coas medianas calculadas en train.
      - Normalización cos parámetros (media, std) calculados en train.

    Limpeza (só sobre train):
      1. Duplicados        -> elimínanse directamente (fix: antes convertíanse
                             en NaN, corrompendo datos válidos).
      2. Valores imposibles -> valores fóra dos rangos/conxuntos -> NaN.
      3. Outliers (toggle) -> valores extremos por IQR (1.5-IQR) -> NaN.
      4. Xestión de NaN   -> filas con moitos NaN elimínanse; o resto impútase
                             coa mediana de train.

    Transformación (train e test):
      5. Inxeniería de variables (ratios financeiros + temporais).
      6. Separación do target.
      7. Eliminación de columnas non útiles (ID_Cliente, Data_Solicitude).
      8. Codigo_Postal -> string (fix: trátase como categórica, non numérica).
      9. One-Hot Encoding das variables categóricas.
     10. Aliñamento train/test.
     11. Normalización N(0,1) cos parámetros de train (toggle).
    """
    print("=" * 60)
    print("  PREPROCESADO DE DATOS")
    print("=" * 60)

    train = train.copy()
    test  = test.copy()

    # ── Pasos 1-4: limpeza xeral ────────────────────────────────────
    train = _eliminar_duplicados(train)
    train = _eliminar_imposibles(train)

    if outliers == 'eliminar':
        train, _ = _eliminar_outliers_iqr(train)
    elif outliers == 'capear':
        train, test = _capear_outliers_iqr(train, test)
    else:
        print("[Paso 3] Detección de outliers desactivada.\n")

    train, medianas = _xestionar_nans_train(train, umbral_nan)

    # ── Imputación do test coas medianas de train ─────────────────────────────
    if imputar_nan_test:
        print("[Test] Imputando NaN coas medianas de train...")
        n_imputadas = test.isna().sum().sum()
        for col in test.columns:
            if test[col].isna().any() and col in medianas.index:
                test[col] = test[col].fillna(medianas[col])
        print(f"  - {n_imputadas} celdas imputadas\n")

    # ── Paso 5: Inxeniería de variables ───────────────────────────────────────
    print("[Paso 5] Creando novas variables derivadas...")
    train = crear_features(train)
    test  = crear_features(test)

    # ── Paso 6: Separar o target ──────────────────────────────────────────────
    y_train = train['Target_Risco'].copy()

    # ── Paso 7: Eliminar columnas non útiles ──────────────────────────────────
    cols_drop_train = ['ID_Cliente', 'Data_Solicitude', 'Target_Risco']
    cols_drop_test  = ['ID_Cliente', 'Data_Solicitude']

    X_train_raw = train.drop(columns=[c for c in cols_drop_train if c in train.columns])
    X_test_raw  = test.drop(columns=[c for c in cols_drop_test  if c in test.columns])

    # ── Paso 8: Codigo_Postal como categórica ─────────────────────────────────
    # [FIX] Código postal só ten 13 valores únicos. Tratar como numérico
    # continuo non ten sentido. Convertemos a string para que o OHE o trate
    # como calquera outra variable categórica.
    for df_ in [X_train_raw, X_test_raw]:
        if 'Codigo_Postal' in df_.columns:
            df_['Codigo_Postal'] = 'CP_' + df_['Codigo_Postal'].astype(int).astype(str)

    # ── Paso 9: One-Hot Encoding ──────────────────────────────────────────────
    X_train_encoded = pd.get_dummies(X_train_raw)
    X_test_encoded  = pd.get_dummies(X_test_raw)

    # ── Paso 10: Aliñamento ───────────────────────────────────────────────────
    X_train_aligned, X_test_aligned = X_train_encoded.align(
        X_test_encoded, join='left', axis=1, fill_value=0
    )

    # ── Paso 11: Transformación logarítmica ─────────────────────────────────
    if trans_log:
        X_train_aligned, n_trans = _aplicar_transformacion_log(X_train_aligned, _COLS_TRANSFORMACION_LOG)
        X_test_aligned, _        = _aplicar_transformacion_log(X_test_aligned, _COLS_TRANSFORMACION_LOG)
        if n_trans > 0:
            print(f"  - {n_trans} variables transformadas para suavizar a distribución.\n")
        else:
            print(f"  - Ningunha variable a transformar.\n")

    # ── Paso 12: Normalización cos parámetros de train ────────────────────────
    if normalizar:
        X_train_aligned, X_test_aligned = _normalizar(X_train_aligned, X_test_aligned)
    else:
        print("[Paso 12] Normalización desactivada.\n")

    print(f"{'=' * 60}")
    print(f"  Preprocesado rematado.")
    print(f"  Train final : {X_train_aligned.shape[0]} filas - {X_train_aligned.shape[1]} variables")
    print(f"  Test final  : {X_test_aligned.shape[0]} filas - {X_test_aligned.shape[1]} variables")
    print(f"{'=' * 60}\n")

    return X_train_aligned, X_test_aligned, y_train


# ─────────────────────────────────────────────────────────────────────────────
# Funcións auxiliares (privadas)
# ─────────────────────────────────────────────────────────────────────────────

def _eliminar_duplicados(train: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina directamente as filas duplicadas de train.

    [FIX] A versión anterior convertía as filas duplicadas en NaN, o que
    corrompía os datos válidos desas filas (incluíndo o target). A corrección
    correcta é eliminar as filas duplicadas directamente con drop_duplicates.
    """
    print("[Paso 1] Eliminando duplicados en train...")
    n_antes = len(train)
    train = train.drop_duplicates(keep='first').reset_index(drop=True)
    n_dup = n_antes - len(train)
    if n_dup > 0:
        print(f"  - {n_dup} filas duplicadas eliminadas")
    else:
        print(f"  - Sen duplicados")
    print()
    return train


def _eliminar_imposibles(train: pd.DataFrame) -> pd.DataFrame:
    """Converte en NaN os valores fóra dos rangos/conxuntos válidos en train."""
    print("[Paso 2] Eliminando valores imposibles en train...")
    total_celdas_nan = 0
    filas_afectadas  = set()

    for col, restricion in _RESTRICCIONS.items():
        if col not in train.columns:
            continue
        if col == 'Data_Solicitude':
            mascara = _mascara_data_imposible(train[col], restricion)
        elif isinstance(restricion, set):
            mascara = _mascara_categorica_imposible(train[col], restricion)
        else:
            mascara = _mascara_numerica_imposible(train[col], restricion)

        n_malos = mascara.sum()
        if n_malos > 0:
            train.loc[mascara, col] = np.nan
            total_celdas_nan += n_malos
            filas_afectadas.update(train.index[mascara].tolist())

    print(f"  - {total_celdas_nan} celdas -> NaN en {len(filas_afectadas)} filas\n")
    return train


def _eliminar_outliers_iqr(
    train: pd.DataFrame,
    factor: float = 1.5,
) -> tuple[pd.DataFrame, dict]:
    """Detecta outliers por IQR en train e convérteos en NaN."""
    print(f"[Paso 3] Detectando outliers en train (IQR - {factor})...")
    limites = {}
    total_celdas_nan = 0
    filas_afectadas  = set()

    for col in _COLS_OUTLIERS:
        if col not in train.columns:
            continue
        q1  = train[col].quantile(0.25)
        q3  = train[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        limites[col] = (lo, hi)

        mascara = train[col].notna() & ((train[col] < lo) | (train[col] > hi))
        n_malos = mascara.sum()
        if n_malos > 0:
            train.loc[mascara, col] = np.nan
            total_celdas_nan += n_malos
            filas_afectadas.update(train.index[mascara].tolist())

    print(f"  - {total_celdas_nan} outliers -> NaN en {len(filas_afectadas)} filas\n")
    return train, limites

# En vez de eliminar a un que teña ingresos multimillonarios, deixalo nun límite superior razonable.
def _capear_outliers_iqr(
    train: pd.DataFrame,
    test: pd.DataFrame,
    factor: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Capa (winsorize) os outliers en train e test usando os límites de train."""
    print(f"[Paso 3] Capeando outliers (Winsorización) con IQR - {factor}...")
    
    train_capeado = train.copy()
    test_capeado = test.copy()
    n_capeados = 0

    for col in _COLS_OUTLIERS:
        if col not in train.columns:
            continue
        
        q1  = train[col].quantile(0.25)
        q3  = train[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr

        # Clip limita os valores por abaixo (lo) e por arriba (hi)
        train_capeado[col] = train_capeado[col].clip(lower=lo, upper=hi)
        
        if col in test_capeado.columns:
            test_capeado[col] = test_capeado[col].clip(lower=lo, upper=hi)
            
        n_capeados += 1

    print(f"  - {n_capeados} variables capeadas aos límites razoables.\n")
    return train_capeado, test_capeado

def _xestionar_nans_train(
    train: pd.DataFrame,
    umbral: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Sobre train:
      - Elimina filas con máis de `umbral` NaN.
      - Imputa o resto coa mediana de train.
    Devolve o train limpo e as medianas (para usarlas despois en test).
    """
    print(f"[Paso 4] Xestionando NaN en train (umbral = {umbral} NaN por fila)...")

    nans_por_fila    = train.isna().sum(axis=1)
    mascara_eliminar = nans_por_fila > umbral
    n_eliminadas     = mascara_eliminar.sum()
    train.drop(index=train.index[mascara_eliminar], inplace=True)

    medianas    = train.median(numeric_only=True)
    n_imputadas = train.isna().sum().sum()
    for col in train.columns:
        if train[col].isna().any():
            train[col] = train[col].fillna(medianas.get(col, 0))

    print(f"  - {n_eliminadas} filas eliminadas - {n_imputadas} celdas imputadas coa mediana\n")
    return train, medianas


def _normalizar(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estandariza a N(0,1) cos parámetros de train. As columnas OHE non se normalizan."""
    print("[Paso 11] Normalizando variables a N(0,1)...")

    X_train_norm = X_train.copy()
    X_test_norm  = X_test.copy()

    cols_onehot    = [col for col in X_train.columns
                      if set(X_train[col].dropna().unique()).issubset({0, 1})]
    cols_numericas = [col for col in X_train.columns if col not in cols_onehot]

    n_normalizadas = 0
    for col in cols_numericas:
        media = X_train[col].mean()
        std   = X_train[col].std()
        if std == 0:
            continue
        X_train_norm[col] = (X_train[col] - media) / std
        X_test_norm[col]  = (X_test[col]  - media) / std
        n_normalizadas += 1

    print(f"  - {n_normalizadas} variables normalizadas - {len(cols_onehot)} OHE omitidas\n")
    return X_train_norm, X_test_norm

def _aplicar_transformacion_log(df: pd.DataFrame, columnas_log: list) -> tuple[pd.DataFrame, int]:
    """
    Aplica a transformación matemática np.log1p ás columnas indicadas para reducir o sesgo de variables con "colas longas" (normalmente monetarias).
    """
    df_trans = df.copy()
    n_transformadas = 0
    
    for col in columnas_log:
        if col in df_trans.columns:
            # clip(lower=0) garante que non entren valores negativos que rompan o logaritmo
            df_trans[col] = np.log1p(df_trans[col].clip(lower=0))
            n_transformadas += 1
            
    return df_trans, n_transformadas


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de máscara
# ─────────────────────────────────────────────────────────────────────────────

def _mascara_numerica_imposible(serie: pd.Series, restricion: tuple) -> pd.Series:
    lo, hi  = restricion
    mascara = pd.Series(False, index=serie.index)
    if lo is not None:
        mascara |= serie.notna() & (serie < lo)
    if hi is not None:
        mascara |= serie.notna() & (serie > hi)
    return mascara

def _mascara_categorica_imposible(serie: pd.Series, valores_validos: set) -> pd.Series:
    return serie.notna() & ~serie.isin(valores_validos)

def _mascara_data_imposible(serie: pd.Series, restricion: tuple) -> pd.Series:
    data_min = pd.Timestamp(restricion[0])
    data_max = pd.Timestamp(restricion[1])
    datas    = pd.to_datetime(serie, errors='coerce')
    return datas.notna() & ((datas < data_min) | (datas > data_max))
