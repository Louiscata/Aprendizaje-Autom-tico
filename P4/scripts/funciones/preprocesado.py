import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Función de selección de variables
# ─────────────────────────────────────────────────────────────────────────────
    
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

import pandas as pd
import numpy as np


# ── Restriccións de dominio ───────────────────────────────────────────────────
#
#   Cada entrada define os valores/rangos válidos para a súa variable.
#   Formato:
#     - Numérico continuo  → (min, max)   [None = sen límite nese extremo]
#     - Conxunto de valores válidos → set(...)
#
# _RESTRICCIONS aplícase ANTES de codificar (usa nomes de columna orixinais).

_RESTRICCIONS: dict = {
    # ── Identificación e data ────────────────────────────────────────────────
    'ID_Cliente':                  (100_000, None),
    'Data_Solicitude':             ('2021-01-01', '2023-12-31'),   # tratada á parte
    # ── Perfil persoal ───────────────────────────────────────────────────────
    'Idade':                       (18, 85),
    'Lonxitude_Nome':              (10, 35),
    'Num_Fillos':                  (0, 12),
    'Profesion':                   {'Asalariado', 'Autónomo', 'Funcionario',
                                    'Estudante', 'Desempregado'},
    'Anos_Emprego':                (0.0, 62.0),
    'Ingresos_Anuais':             (0, None),
    # ── Comportamento web ────────────────────────────────────────────────────
    'Tipo_Dispositivo':            {'PC_Windows', 'Mac', 'iPhone', 'Android', 'Linux'},
    'Tempo_Web_Minutos':           (1, 50),
    'Subscricion_Email':           {0, 1},
    'Dia_Solicitude':              {'Luns', 'Martes', 'Mércores', 'Xoves',
                                    'Venres', 'Sábado', 'Domingo'},
    # ── Xeografía e patrimonio ───────────────────────────────────────────────
    'Distancia_Oficina_Km':        (0, None),
    'Codigo_Postal':               (0, None),
    'Patrimonio_Total':            (0, None),
    'Debeda_Total':                (0, None),
    # ── Historial crediticio ─────────────────────────────────────────────────
    'Historial_Impagos':           {0, 1},
    'Numero_Tarxetas':             (0, 12),
    'Utilizacion_Credito':         (0.0, 1.2),
    'Consultas_Risco_6M':          (0, 17),
    'Limite_Credito_Total':        (0, None),
    'Cota_Mensual_Prestamos':      (0, None),
    'Ratio_Cota_Ingresos':         (0.0, 2.5),
    'Prestamos_Activos':           (0, 9),
    'Antiguedade_Cliente_Anos':    (0.0, 40.0),
    # ── Saldo e estres ───────────────────────────────────────────────────────
    'Saldo_Medio_3M':              (0, None),
    'Variacion_Saldo_6M':          (-1.0, 1.0),
    'Fondo_Emerxencia_Meses':      (0.0, 24.0),
    'Indice_Estres_Financeiro':    (0.0, 2.0),
    # ── Target (só en train) ─────────────────────────────────────────────────
    'Target_Risco':                {0, 1, 2, 3},
}

# Columnas numéricas continuas sobre as que aplicar detección de outliers IQR.
# Excluímos variables binarias, discretas de rango moi pequeno e identificadores,
# porque os seus valores posibles xa quedan cubertos polas restriccións de dominio.
_COLS_OUTLIERS = [
    'Idade', 'Lonxitude_Nome', 'Anos_Emprego', 'Ingresos_Anuais',
    'Tempo_Web_Minutos', 'Distancia_Oficina_Km', 'Patrimonio_Total',
    'Debeda_Total', 'Utilizacion_Credito', 'Limite_Credito_Total',
    'Cota_Mensual_Prestamos', 'Ratio_Cota_Ingresos', 'Antiguedade_Cliente_Anos',
    'Saldo_Medio_3M', 'Variacion_Saldo_6M', 'Fondo_Emerxencia_Meses',
    'Indice_Estres_Financeiro',
]


# ─────────────────────────────────────────────────────────────────────────────
# Función de preprocesado
# ─────────────────────────────────────────────────────────────────────────────

def preprocesar_datos(
    train: pd.DataFrame,
    test: pd.DataFrame,
    umbral_nan: int = 10,
    eliminar_outliers: bool = True,
    imputar_nan_test: bool = True,
    normalizar: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Preprocesa os datos de adestramento e test.

    O test NON se limpa: non se eliminan duplicados, valores imposibles,
    outliers nin filas con moitos NaN. O test só recibe:
      · Imputación de NaN coas medianas calculadas en train.
      · Normalización cos parámetros (media, std) calculados en train.

    Limpeza (só sobre train):
      1. Duplicados        → filas duplicadas convértense en NaN.
      2. Valores imposibles → valores fóra dos rangos/conxuntos válidos → NaN.
      3. Outliers (toggle) → valores extremos por IQR (1.5·IQR) → NaN.
      4. Xestión de NaN   → filas con moitos NaN elimínanse; o resto impútase
                             coa mediana de train.

    Transformación (train e test):
      5. Separación do target.
      6. Eliminación de columnas non útiles (ID_Cliente, Data_Solicitude).
      7. One-Hot Encoding das variables categóricas.
      8. Aliñamento train/test.
      9. Normalización N(0,1) cos parámetros de train (toggle).
    """
    print("=" * 55)
    print("  PREPROCESADO DE DATOS")
    print("=" * 55)

    train = train.copy()
    test  = test.copy()

    # ── Pasos 1-4: limpeza só sobre train ────────────────────────────────────
    train = _eliminar_duplicados(train)
    train = _eliminar_imposibles(train)

    if eliminar_outliers:
        train, limites_iqr = _eliminar_outliers_iqr(train)
    else:
        print("[Paso 3] Detección de outliers desactivada.\n")
        limites_iqr = {}

    train, medianas = _xestionar_nans_train(train, umbral_nan)

    # ── Imputación do test coas medianas de train ─────────────────────────────
    if(imputar_nan_test):
        print("[Test] Imputando NaN coas medianas de train...")
        n_imputadas = test.isna().sum().sum()
        for col in test.columns:
            if test[col].isna().any() and col in medianas.index:
                test[col] = test[col].fillna(medianas[col])
        print(f"  · {n_imputadas} celdas imputadas\n")

    # ── Paso 5: Separar o target ──────────────────────────────────────────────
    y_train = train['Target_Risco'].copy()

    # ── Paso 6: Eliminar columnas non útiles ──────────────────────────────────
    cols_drop_train = ['ID_Cliente', 'Target_Risco', 'Data_Solicitude']
    cols_drop_test  = ['ID_Cliente', 'Data_Solicitude']

    X_train_raw = train.drop(columns=[c for c in cols_drop_train if c in train.columns])
    X_test_raw  = test.drop(columns=[c for c in cols_drop_test  if c in test.columns])

    # ── Paso 7: One-Hot Encoding ──────────────────────────────────────────────
    X_train_encoded = pd.get_dummies(X_train_raw)
    X_test_encoded  = pd.get_dummies(X_test_raw)

    # ── Paso 8: Aliñamento ────────────────────────────────────────────────────
    X_train_aligned, X_test_aligned = X_train_encoded.align(
        X_test_encoded, join='left', axis=1, fill_value=0
    )

    # ── Paso 9: Normalización cos parámetros de train ─────────────────────────
    if normalizar:
        X_train_aligned, X_test_aligned = _normalizar(X_train_aligned, X_test_aligned)
    else:
        print("[Paso 9] Normalización desactivada.\n")

    print(f"{'=' * 55}")
    print(f"  Preprocesado rematado.")
    print(f"  Train final : {X_train_aligned.shape[0]} filas · {X_train_aligned.shape[1]} variables")
    print(f"  Test final  : {X_test_aligned.shape[0]} filas · {X_test_aligned.shape[1]} variables")
    print(f"{'=' * 55}\n")

    return X_train_aligned, X_test_aligned, y_train


# ─────────────────────────────────────────────────────────────────────────────
# Funcións auxiliares (privadas) — só actúan sobre train
# ─────────────────────────────────────────────────────────────────────────────

def _eliminar_duplicados(train: pd.DataFrame) -> pd.DataFrame:
    """Converte en NaN as filas duplicadas de train (mantén a primeira ocorrencia)."""
    print("[Paso 1] Eliminando duplicados en train...")
    mascara_dup = train.duplicated(keep='first')
    n_dup = mascara_dup.sum()
    if n_dup > 0:
        train.loc[mascara_dup, :] = np.nan
        print(f"  · Train: {n_dup} filas duplicadas → convertidas en NaN")
    else:
        print(f"  · Train: sen duplicados")
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

    print(f"  · Train: {total_celdas_nan} celdas → NaN en {len(filas_afectadas)} filas\n")
    return train


def _eliminar_outliers_iqr(
    train: pd.DataFrame,
    factor: float = 1.5,
) -> tuple[pd.DataFrame, dict]:
    """
    Detecta outliers por IQR en train e convérteos en NaN.
    Devolve o train limpo e os límites calculados (por se se necesitan noutro lado).
    """
    print(f"[Paso 3] Detectando outliers en train (IQR · {factor})...")
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

    print(f"  · Train: {total_celdas_nan} outliers → NaN en {len(filas_afectadas)} filas\n")
    return train, limites


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

    medianas     = train.median(numeric_only=True)
    n_imputadas  = train.isna().sum().sum()
    for col in train.columns:
        if train[col].isna().any():
            train[col] = train[col].fillna(medianas.get(col, 0))

    print(f"  · Train: {n_eliminadas} filas eliminadas · {n_imputadas} celdas imputadas coa mediana\n")
    return train, medianas


def _normalizar(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estandariza a N(0,1) cos parámetros de train. As columnas one-hot non se normalizan."""
    print("[Paso 9] Normalizando variables a N(0,1)...")

    X_train_norm = X_train.copy()
    X_test_norm  = X_test.copy()

    cols_onehot    = [col for col in X_train.columns
                      if X_train[col].dropna().isin([0, 1]).all()]
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

    print(f"  · {n_normalizadas} variables normalizadas · {len(cols_onehot)} one-hot omitidas\n")
    return X_train_norm, X_test_norm


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