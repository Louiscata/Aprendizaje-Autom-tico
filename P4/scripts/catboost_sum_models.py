"""
catboost_sum_models.py
──────────────────────
Competición Kaggle – Clasificación de risco crediticio (Target_Risco: 0-3)

Estratexia:
  1. Preprocesado: eliminación de duplicados + imputación por mediana
  2. OHE para variables sen sentido numérico (Profesion, Tipo_Dispositivo,
     Dia_Solicitude, Codigo_Postal)
  3. Selección das top-17 variables por importancia de CatBoost
  4. Para cada modelo dos CONFIGS:
       a. Validación temporal con TimeSeriesSplit (train=pasado, val=futuro)
       b. Adestramento final sobre todo o train
  5. Combinación dos modelos finais con catboost.sum_models
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool, sum_models
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

# ── 0. Configuración ─────────────────────────────────────────────────────────
SEED       = 777
CV_FOLDS   = 5
N_FEATURES = 17
RUTA_TRAIN = './data/train.csv'
RUTA_TEST  = './data/test.csv'

# Modelos que se van a validar E despois combinar con sum_models.
# A diversidade conséguese variando semente, profundidade e learning rate.
CONFIGS = [
    dict(iterations=1200, learning_rate=0.03, depth=7, random_seed=777),
    dict(iterations=1200, learning_rate=0.03, depth=7, random_seed=42),
    dict(iterations=1000, learning_rate=0.04, depth=8, random_seed=123),
    dict(iterations=1000, learning_rate=0.04, depth=6, random_seed=999),
    dict(iterations=1400, learning_rate=0.02, depth=7, random_seed=2024),
]

PARAMS_COMUNS = dict(
    loss_function='MultiClass',
    eval_metric='TotalF1',
    auto_class_weights='Balanced',
    verbose=0,
    thread_count=-1,
)

# ── 1. Carga de datos ────────────────────────────────────────────────────────
print("=" * 60)
print("  CARGA DE DATOS")
print("=" * 60)
train = pd.read_csv(RUTA_TRAIN)
test  = pd.read_csv(RUTA_TEST)
print(f"  Train bruto : {train.shape[0]} filas · {train.shape[1]} columnas")
print(f"  Test  bruto : {test.shape[0]} filas · {test.shape[1]} columnas\n")

# ── 2. Preprocesado ──────────────────────────────────────────────────────────
print("=" * 60)
print("  PREPROCESADO")
print("=" * 60)

n_antes = len(train)
train = train.drop_duplicates(keep='first').reset_index(drop=True)
print(f"  Duplicados eliminados : {n_antes - len(train)}")

train['Data_Solicitude'] = pd.to_datetime(train['Data_Solicitude'])
test['Data_Solicitude']  = pd.to_datetime(test['Data_Solicitude'])
train = train.sort_values('Data_Solicitude').reset_index(drop=True)
print(f"  Rango temporal train  : {train['Data_Solicitude'].min().date()}  →  "
      f"{train['Data_Solicitude'].max().date()}")

y_train     = train['Target_Risco'].copy()
ids_test    = test['ID_Cliente'].copy()
dates_train = train['Data_Solicitude'].copy()  # só para splits temporais, non como feature

X_raw = train.drop(columns=['ID_Cliente', 'Data_Solicitude', 'Target_Risco'])
T_raw = test.drop(columns=['ID_Cliente', 'Data_Solicitude'])

print("  Imputando NaN...")
for col in X_raw.columns:
    if X_raw[col].dtype == object:
        fill = X_raw[col].mode()[0]
    else:
        fill = X_raw[col].median()
    X_raw[col] = X_raw[col].fillna(fill)
    T_raw[col] = T_raw[col].fillna(fill)

# ── 3. One-Hot Encoding ───────────────────────────────────────────────────────
# Variables convertidas a OHE e razón:
#   Profesion        → categorías nominais sen orde
#   Tipo_Dispositivo → SO sen relación de magnitude entre valores
#   Dia_Solicitude   → día da semana con nomes, sen orde cardinal
#   Codigo_Postal    → código xeográfico; só 13 valores únicos;
#                      a diferenza aritmética entre códigos non ten significado
# Mantéñense numéricas:
#   Subscricion_Email, Historial_Impagos → binarios 0/1
#   Num_Fillos, Numero_Tarxetas, Prestamos_Activos, Consultas_Risco_6M → conteos
OHE_COLS = ['Profesion', 'Tipo_Dispositivo', 'Dia_Solicitude', 'Codigo_Postal']

print(f"\n  Aplicando OHE a: {OHE_COLS}")
for col in OHE_COLS:
    X_raw[col] = col + '_' + X_raw[col].astype(str)
    T_raw[col] = col + '_' + T_raw[col].astype(str)

X_enc = pd.get_dummies(X_raw, columns=OHE_COLS, dtype=int)
T_enc = pd.get_dummies(T_raw, columns=OHE_COLS, dtype=int)
X_enc, T_enc = X_enc.align(T_enc, join='left', axis=1, fill_value=0)

print(f"  Variables tras OHE  : {X_enc.shape[1]}")
print(f"  Train final         : {X_enc.shape}\n")

# ── 4. Selección das top-{N_FEATURES} variables (importancia CatBoost) ───────
print("=" * 60)
print(f"  SELECCIÓN DE VARIABLES  (top-{N_FEATURES} por importancia CatBoost)")
print("=" * 60)

aux_model = CatBoostClassifier(
    iterations=400, learning_rate=0.08, depth=6,
    loss_function='MultiClass', eval_metric='TotalF1',
    auto_class_weights='Balanced', random_seed=SEED,
    verbose=0, thread_count=-1,
)
aux_model.fit(Pool(X_enc, label=y_train))

importances = pd.Series(
    aux_model.get_feature_importance(Pool(X_enc, label=y_train)),
    index=X_enc.columns
).sort_values(ascending=False)

FEATURES = importances.index[:N_FEATURES].tolist()
print(f"\n  Top-{N_FEATURES} variables:")
for i, feat in enumerate(FEATURES, 1):
    print(f"    {i:>2}. {feat:<42}  imp: {importances[feat]:.4f}")
print()

X = X_enc[FEATURES].copy()
T = T_enc[FEATURES].copy()

# ── 5. Validación + adestramento final de cada modelo ────────────────────────
# Para cada configuración:
#   a) Validación temporal con TimeSeriesSplit: mide o F1-macro real
#      (sempre trainando co pasado e validando co futuro)
#   b) Adestramento final sobre todo o train co mesmo config
# Desta forma o que se valida e o que se combina son exactamente o mesmo.
print("=" * 60)
print(f"  VALIDACIÓN TEMPORAL + ADESTRAMENTO FINAL")
print(f"  ({len(CONFIGS)} modelos · TimeSeriesSplit · {CV_FOLDS} folds)")
print("=" * 60)

tscv      = TimeSeriesSplit(n_splits=CV_FOLDS)
full_pool = Pool(X, label=y_train)
modelos   = []
f1_por_config = []

for i, cfg in enumerate(CONFIGS, 1):
    desc = (f"iter={cfg['iterations']}, lr={cfg['learning_rate']}, "
            f"depth={cfg['depth']}, seed={cfg['random_seed']}")
    print(f"\n  ── Modelo {i}/{len(CONFIGS)}  ({desc})")

    # a) Validación temporal
    f1_folds = []
    for fold, (idx_tr, idx_val) in enumerate(tscv.split(X), 1):
        X_tr, y_tr   = X.iloc[idx_tr],  y_train.iloc[idx_tr]
        X_val, y_val = X.iloc[idx_val], y_train.iloc[idx_val]

        date_tr_max  = dates_train.iloc[idx_tr].max().date()
        date_val_min = dates_train.iloc[idx_val].min().date()

        m_cv = CatBoostClassifier(**cfg, **PARAMS_COMUNS, early_stopping_rounds=50)
        m_cv.fit(Pool(X_tr, label=y_tr), eval_set=Pool(X_val, label=y_val))

        preds = m_cv.predict(X_val).ravel().astype(int)
        f1    = f1_score(y_val, preds, average='macro')
        f1_folds.append(f1)
        print(f"    Fold {fold}  |  train ≤ {date_tr_max}  |  "
              f"val ≥ {date_val_min}  |  F1-macro = {f1:.5f}")

    f1_medio = float(np.mean(f1_folds))
    f1_por_config.append(f1_medio)
    print(f"    → F1-macro medio : {f1_medio:.5f}  (std: {np.std(f1_folds):.5f})")

    # b) Adestramento final sobre todo o train (mesmo config, sen early stopping)
    print(f"    Adestramento final sobre train completo...", end=' ', flush=True)
    m_final = CatBoostClassifier(**cfg, **PARAMS_COMUNS)
    m_final.fit(full_pool)
    modelos.append(m_final)
    print("listo.")

# Resumo da validación
print(f"\n{'─' * 60}")
print("  RESUMO DE VALIDACIÓN:")
for i, (cfg, f1) in enumerate(zip(CONFIGS, f1_por_config), 1):
    print(f"    Modelo {i}  (seed={cfg['random_seed']}, depth={cfg['depth']})  "
          f"→  F1-macro CV = {f1:.5f}")
print(f"    Media xeral : {np.mean(f1_por_config):.5f}")
print(f"{'─' * 60}\n")

# ── 6. Combinación con sum_models ────────────────────────────────────────────
print("=" * 60)
print(f"  COMBINACIÓN  (sum_models · {len(modelos)} modelos · pesos uniformes)")
print("=" * 60)

pesos        = [1.0 / len(modelos)] * len(modelos)
modelo_final = sum_models(modelos, weights=pesos)
print("  Combinación completada.\n")

# ── 7. Predición e arquivo de envío ──────────────────────────────────────────
print("=" * 60)
print("  PREDICIÓN E EXPORTACIÓN")
print("=" * 60)

preds_test = modelo_final.predict_proba(Pool(T)).argmax(axis=1).astype(int)

submission = pd.DataFrame({
    'ID_Cliente':   ids_test,
    'Target_Risco': preds_test,
})

nome_arquivo = f'./resultados/27-04-2026_catboost_summodels_top{N_FEATURES}.csv'
submission.to_csv(nome_arquivo, index=False)

print(f"  Arquivo xerado : {nome_arquivo}  ({len(submission)} filas)")
print(f"  Distribución de clases preditas:")
print(submission['Target_Risco'].value_counts().sort_index().to_string(header=False))
print("\n✔ Proceso completado con éxito.")
