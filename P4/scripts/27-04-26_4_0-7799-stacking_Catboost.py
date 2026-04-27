"""
stacking_catboost_oof.py
────────────────────────
Competición Kaggle – Clasificación de risco crediticio (Target_Risco: 0-3)

Estratexia:
  1. Preprocesado do compañeiro: eliminación de duplicados, variables lixo,
     enxeñaría de variables financeiras, imputación por mediana,
     categóricas nativas de CatBoost (sen OHE)
  2. Selección das top-N_FEATURES variables por importancia de CatBoost
  3. Walk-Forward CV temporal (expanding window):
       - Train = todo o anterior ao corte de data
       - Val   = VAL_MESES seguintes ao corte
       - Garante que sempre se valida co futuro e se adestra co pasado
  4. Prediccións OOF (Out-Of-Fold) para construír o meta-learner sen data leakage
  5. Meta-learner (Regresión Loxística) que aprende a combinar as probabilidades
     dos modelos base
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

# ── 0. Configuración ─────────────────────────────────────────────────────────
SEED       = 777
N_FEATURES = 11
RUTA_TRAIN = './data/train.csv'
RUTA_TEST  = './data/test.csv'
RUTA_OUT   = './resultados/27-04-2026_4_stacking-oof.csv'

CORTES_MESES = [12, 16, 20]
VAL_MESES    = 4

CONFIGS = [
    dict(iterations=1200, learning_rate=0.03, depth=7, random_seed=777),
    dict(iterations=1000, learning_rate=0.04, depth=8, random_seed=42),
    dict(iterations=1000, learning_rate=0.04, depth=6, random_seed=123),
    dict(iterations=1400, learning_rate=0.02, depth=7, random_seed=999),
    dict(iterations=1200, learning_rate=0.03, depth=5, random_seed=2024),
]

PARAMS_COMUNS = dict(
    loss_function='MultiClass',
    eval_metric='TotalF1',
    auto_class_weights='Balanced',
    verbose=0,
    thread_count=-1,
)

# ── Constantes de preprocesado (do compañeiro) ────────────────────────────────
MALARDAS   = ['ID_Cliente', 'Lonxitude_Nome', 'Tempo_Web_Minutos', 'Subscricion_Email']
CATEGORICAS = ['Profesion', 'Tipo_Dispositivo', 'Dia_Solicitude', 'Codigo_Postal']

# ── Funcións de preprocesado (do compañeiro) ──────────────────────────────────
def crear_features(df_input):
    df  = df_input.copy()
    eps = 0.1

    df['ratio_debeda_ingresos'] = df['Debeda_Total'] / (df['Ingresos_Anuais'] + eps)
    df['patrimonio_neto']       = df['Patrimonio_Total'] - df['Debeda_Total']

    ingreso_mensual             = df['Ingresos_Anuais'] / 12 + eps
    df['ratio_saldo_mensual']   = df['Saldo_Medio_3M'] / ingreso_mensual

    df['ratio_limite_ingresos'] = df['Limite_Credito_Total'] / (df['Ingresos_Anuais'] + eps)

    patrimonio_neto_pos         = df['patrimonio_neto'].clip(lower=eps)
    df['ratio_cota_patrimonio'] = df['Cota_Mensual_Prestamos'] / patrimonio_neto_pos

    return df


def preprocesar_datos(train_raw, test_raw):
    train = train_raw.copy()
    test  = test_raw.copy()

    # 1. Duplicados
    n_antes = len(train)
    train   = train.drop_duplicates(keep='first').reset_index(drop=True)
    if (n_antes - len(train)) > 0:
        print(f"  -> Eliminados {n_antes - len(train)} duplicados.")

    # 2. Enxeñaría de variables
    train = crear_features(train)
    test  = crear_features(test)

    # 3. Target e datas (gardamos as datas para os splits temporais)
    y_train     = train['Target_Risco'].copy()
    dates_train = pd.to_datetime(train['Data_Solicitude'])
    ids_test    = test['ID_Cliente'].copy()

    # 4. Eliminar variables lixo + target + data
    cols_a_borrar = MALARDAS + ['Target_Risco', 'Data_Solicitude']
    X_train = train.drop(columns=[c for c in cols_a_borrar if c in train.columns])
    X_test  = test.drop(columns=[c for c in cols_a_borrar if c in test.columns])

    # 5. Categóricas: prefixar Codigo_Postal e imputar nulos
    for col in CATEGORICAS:
        if col in X_train.columns:
            if col == 'Codigo_Postal':
                X_train[col] = 'CP_' + X_train[col].astype(str)
                X_test[col]  = 'CP_' + X_test[col].astype(str)
            X_train[col] = X_train[col].fillna('DESCONECIDO').astype(str)
            X_test[col]  = X_test[col].fillna('DESCONECIDO').astype(str)

    # 6. Numéricas: imputar con mediana
    cols_numericas = [c for c in X_train.columns if c not in CATEGORICAS]
    for col in cols_numericas:
        if X_train[col].dtype in [np.float64, np.int64]:
            mediana     = X_train[col].median()
            X_train[col] = X_train[col].fillna(mediana)
            X_test[col]  = X_test[col].fillna(mediana)

    print(f"  -> OHE desactivado. Preparado para motor CatBoost nativo.")
    print(f"  -> Variables finais xeradas: {X_train.shape[1]}")

    return X_train, X_test, y_train, dates_train, ids_test


# ── 1. Carga de datos ────────────────────────────────────────────────────────
print("=" * 60)
print("  CARGA DE DATOS")
print("=" * 60)
train_raw = pd.read_csv(RUTA_TRAIN)
test_raw  = pd.read_csv(RUTA_TEST)
print(f"  Train bruto : {train_raw.shape[0]} filas · {train_raw.shape[1]} columnas")
print(f"  Test  bruto : {test_raw.shape[0]} filas · {test_raw.shape[1]} columnas\n")

# ── 2. Preprocesado ──────────────────────────────────────────────────────────
print("=" * 60)
print("  PREPROCESADO")
print("=" * 60)

train_raw['Data_Solicitude'] = pd.to_datetime(train_raw['Data_Solicitude'])
test_raw['Data_Solicitude']  = pd.to_datetime(test_raw['Data_Solicitude'])
train_raw = train_raw.sort_values('Data_Solicitude').reset_index(drop=True)

print(f"  Rango temporal train  : {train_raw['Data_Solicitude'].min().date()}  →  "
      f"{train_raw['Data_Solicitude'].max().date()}")
print(f"  Rango temporal test   : {test_raw['Data_Solicitude'].min().date()}  →  "
      f"{test_raw['Data_Solicitude'].max().date()}")

X_raw, T_raw, y_train, dates_train, ids_test = preprocesar_datos(train_raw, test_raw)

fecha_min = dates_train.min()
cat_idx   = [X_raw.columns.tolist().index(c) for c in CATEGORICAS if c in X_raw.columns]

# ── 3. Selección das top-{N_FEATURES} variables ──────────────────────────────
print("\n" + "=" * 60)
print(f"  SELECCIÓN DE VARIABLES  (top-{N_FEATURES} por importancia CatBoost)")
print("=" * 60)

aux_pool  = Pool(X_raw, label=y_train, cat_features=cat_idx)
aux_model = CatBoostClassifier(
    iterations=400, learning_rate=0.08, depth=6,
    loss_function='MultiClass', eval_metric='TotalF1',
    auto_class_weights='Balanced', random_seed=SEED,
    verbose=0, thread_count=-1,
)
aux_model.fit(aux_pool)

importances = pd.Series(
    aux_model.get_feature_importance(aux_pool),
    index=X_raw.columns
).sort_values(ascending=False)

FEATURES = importances.index[:N_FEATURES].tolist()
print(f"\n  Top-{N_FEATURES} variables:")
for i, feat in enumerate(FEATURES, 1):
    print(f"    {i:>2}. {feat:<42}  imp: {importances[feat]:.4f}")
print()

X             = X_raw[FEATURES].copy()
T             = T_raw[FEATURES].copy()
cat_idx_feat  = [FEATURES.index(c) for c in CATEGORICAS if c in FEATURES]

# ── 4. Definición dos folds temporais ────────────────────────────────────────
print("=" * 60)
print(f"  WALK-FORWARD CV  (expanding window · {len(CORTES_MESES)} folds)")
print(f"  Train = pasado ao corte  |  Val = {VAL_MESES} meses seguintes")
print("=" * 60)

folds = []
for m in CORTES_MESES:
    corte   = fecha_min + pd.DateOffset(months=m)
    val_fin = corte + pd.DateOffset(months=VAL_MESES)
    idx_tr  = dates_train.index[dates_train < corte].tolist()
    idx_val = dates_train.index[(dates_train >= corte) & (dates_train < val_fin)].tolist()
    folds.append((idx_tr, idx_val))
    print(f"  Corte mes {m:>2}  |  train ≤ {corte.date()}  ({len(idx_tr):>5} mostras)  |  "
          f"val {corte.date()} → {val_fin.date()}  ({len(idx_val):>4} mostras)")
print()

# ── 5. OOF stacking ──────────────────────────────────────────────────────────
print("=" * 60)
print(f"  OOF STACKING  ({len(CONFIGS)} modelos base × {len(folds)} folds)")
print("=" * 60)

n_classes = y_train.nunique()
n_models  = len(CONFIGS)

oof_preds  = np.full((len(X), n_models * n_classes), np.nan)
test_preds = np.zeros((n_models, len(folds), len(T), n_classes))
f1_por_config = []

for m_idx, cfg in enumerate(CONFIGS, 1):
    desc = (f"iter={cfg['iterations']}, lr={cfg['learning_rate']}, "
            f"depth={cfg['depth']}, seed={cfg['random_seed']}")
    print(f"\n  ── Modelo {m_idx}/{n_models}  ({desc})")

    f1_folds = []
    for fold, (idx_tr, idx_val) in enumerate(folds, 1):
        X_tr, y_tr   = X.iloc[idx_tr],  y_train.iloc[idx_tr]
        X_val, y_val = X.iloc[idx_val], y_train.iloc[idx_val]

        date_tr_max  = dates_train.iloc[idx_tr].max().date()
        date_val_min = dates_train.iloc[idx_val].min().date()

        m_cv = CatBoostClassifier(**cfg, **PARAMS_COMUNS, early_stopping_rounds=50)
        m_cv.fit(
            Pool(X_tr, label=y_tr, cat_features=cat_idx_feat),
            eval_set=Pool(X_val, label=y_val, cat_features=cat_idx_feat),
        )

        proba_val = m_cv.predict_proba(X_val)
        col_ini   = (m_idx - 1) * n_classes
        col_fin   = m_idx * n_classes
        oof_preds[idx_val, col_ini:col_fin] = proba_val
        test_preds[m_idx - 1, fold - 1]    = m_cv.predict_proba(Pool(T, cat_features=cat_idx_feat))

        f1 = f1_score(y_val, proba_val.argmax(axis=1), average='macro')
        f1_folds.append(f1)
        print(f"    Fold {fold}  |  train ≤ {date_tr_max}  |  "
              f"val ≥ {date_val_min}  |  F1-macro = {f1:.5f}")

    f1_medio = float(np.mean(f1_folds))
    f1_por_config.append(f1_medio)
    print(f"    → F1-macro medio : {f1_medio:.5f}  (std: {np.std(f1_folds):.5f})")

print(f"\n{'─' * 60}")
print("  RESUMO DE VALIDACIÓN (modelos base):")
for i, (cfg, f1) in enumerate(zip(CONFIGS, f1_por_config), 1):
    print(f"    Modelo {i}  (seed={cfg['random_seed']}, depth={cfg['depth']})  "
          f"→  F1-macro CV = {f1:.5f}")
print(f"    Media xeral : {np.mean(f1_por_config):.5f}")
print(f"{'─' * 60}\n")

# ── 6. Meta-learner ───────────────────────────────────────────────────────────
print("=" * 60)
print("  META-LEARNER  (Regresión Loxística sobre probabilidades OOF)")
print("=" * 60)

mask     = ~np.isnan(oof_preds[:, 0])
oof_eval = oof_preds[mask]
y_eval   = y_train[mask]

mean_oof = np.zeros((mask.sum(), n_classes))
for m in range(n_models):
    mean_oof += oof_eval[:, m * n_classes:(m + 1) * n_classes]
mean_oof /= n_models
f1_media = f1_score(y_eval, mean_oof.argmax(axis=1), average='macro')
print(f"  F1-macro OOF media simple  : {f1_media:.5f}")

meta = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED,
                           solver='lbfgs', class_weight='balanced')
meta.fit(oof_eval, y_eval)
f1_meta = f1_score(y_eval, meta.predict(oof_eval), average='macro')
print(f"  F1-macro OOF meta-learner  : {f1_meta:.5f}")

test_meta = np.hstack([test_preds[m].mean(axis=0) for m in range(n_models)])

if f1_meta >= f1_media:
    print("  → Usando meta-learner (mellora ou iguala a media simple)")
    final_preds = meta.predict(test_meta).astype(int)
    metodo = 'meta-learner'
else:
    print("  → Usando media simple (meta-learner non mellora)")
    mean_test = np.zeros((len(T), n_classes))
    for m in range(n_models):
        mean_test += test_preds[m].mean(axis=0)
    mean_test /= n_models
    final_preds = mean_test.argmax(axis=1).astype(int)
    metodo = 'media simple'

# ── 7. Predición e arquivo de envío ──────────────────────────────────────────
print("\n" + "=" * 60)
print("  PREDICIÓN E EXPORTACIÓN")
print("=" * 60)

submission = pd.DataFrame({'ID_Cliente': ids_test, 'Target_Risco': final_preds})
submission.to_csv(RUTA_OUT, index=False)

print(f"  Método usado   : {metodo}")
print(f"  Arquivo xerado : {RUTA_OUT}  ({len(submission)} filas)")
print(f"  Distribución de clases preditas:")
print(submission['Target_Risco'].value_counts().sort_index().to_string(header=False))
print("\n✔ Proceso completado con éxito.")