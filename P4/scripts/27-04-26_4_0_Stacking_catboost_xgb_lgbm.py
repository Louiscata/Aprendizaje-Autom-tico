"""
stacking_catboost_xgb_lgbm.py
──────────────────────────────
Competición Kaggle – Clasificación de risco crediticio (Target_Risco: 0-3)

Estratexia:
  1. Preprocesado do compañeiro: eliminación de duplicados, variables lixo,
     enxeñaría de variables financeiras, imputación por mediana
  2. Dous datasets:
       - Sen OHE → para CatBoost (categóricas nativas)
       - Con OHE → para XGBoost e LightGBM
  3. Selección das top-N_FEATURES variables por importancia (CatBoost auxiliar)
  4. Walk-Forward CV temporal (expanding window)
  5. OOF stacking: 2 CatBoost + 2 XGBoost + 2 LightGBM
  6. Meta-learner (Regresión Loxística)
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

# ── 0. Configuración ─────────────────────────────────────────────────────────
SEED        = 777
N_FEATURES  = 11
RUTA_TRAIN  = './data/train.csv'
RUTA_TEST   = './data/test.csv'
RUTA_OUT    = './resultados/27-04-2026_4_stacking-hetero.csv'

CORTES_MESES = [12, 16, 20]
VAL_MESES    = 4

# ── Constantes de preprocesado (do compañeiro) ────────────────────────────────
MALARDAS    = ['ID_Cliente', 'Lonxitude_Nome', 'Tempo_Web_Minutos', 'Subscricion_Email']
CATEGORICAS = ['Profesion', 'Tipo_Dispositivo', 'Dia_Solicitude', 'Codigo_Postal']

# ── Modelos base ──────────────────────────────────────────────────────────────
CONFIGS_CAT = [
    dict(iterations=1200, learning_rate=0.03, depth=7, random_seed=777),
    dict(iterations=1000, learning_rate=0.04, depth=8, random_seed=42),
]

CONFIGS_XGB = [
    dict(n_estimators=800, learning_rate=0.03, max_depth=7, random_state=777),
    dict(n_estimators=600, learning_rate=0.05, max_depth=6, random_state=42),
]

CONFIGS_LGBM = [
    dict(n_estimators=800, learning_rate=0.03, max_depth=7, random_state=777),
    dict(n_estimators=600, learning_rate=0.05, max_depth=6, random_state=42),
]

PARAMS_CAT = dict(
    loss_function='MultiClass',
    eval_metric='TotalF1',
    auto_class_weights='Balanced',
    verbose=0,
    thread_count=-1,
)

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

    n_antes = len(train)
    train   = train.drop_duplicates(keep='first').reset_index(drop=True)
    if (n_antes - len(train)) > 0:
        print(f"  -> Eliminados {n_antes - len(train)} duplicados.")

    train = crear_features(train)
    test  = crear_features(test)

    y_train     = train['Target_Risco'].copy()
    dates_train = pd.to_datetime(train['Data_Solicitude'])
    ids_test    = test['ID_Cliente'].copy()

    cols_a_borrar = MALARDAS + ['Target_Risco', 'Data_Solicitude']
    X_train = train.drop(columns=[c for c in cols_a_borrar if c in train.columns])
    X_test  = test.drop(columns=[c for c in cols_a_borrar if c in test.columns])

    for col in CATEGORICAS:
        if col in X_train.columns:
            if col == 'Codigo_Postal':
                X_train[col] = 'CP_' + X_train[col].astype(str)
                X_test[col]  = 'CP_' + X_test[col].astype(str)
            X_train[col] = X_train[col].fillna('DESCONECIDO').astype(str)
            X_test[col]  = X_test[col].fillna('DESCONECIDO').astype(str)

    cols_numericas = [c for c in X_train.columns if c not in CATEGORICAS]
    for col in cols_numericas:
        if X_train[col].dtype in [np.float64, np.int64]:
            mediana      = X_train[col].median()
            X_train[col] = X_train[col].fillna(mediana)
            X_test[col]  = X_test[col].fillna(mediana)

    # Dataset con OHE para XGBoost e LightGBM
    X_train_ohe = pd.get_dummies(X_train, columns=CATEGORICAS, dtype=int)
    X_test_ohe  = pd.get_dummies(X_test,  columns=CATEGORICAS, dtype=int)
    X_train_ohe, X_test_ohe = X_train_ohe.align(X_test_ohe, join='left', axis=1, fill_value=0)

    print(f"  -> Variables sen OHE : {X_train.shape[1]}")
    print(f"  -> Variables con OHE : {X_train_ohe.shape[1]}")

    return X_train, X_test, X_train_ohe, X_test_ohe, y_train, dates_train, ids_test


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

X_raw, T_raw, X_ohe, T_ohe, y_train, dates_train, ids_test = preprocesar_datos(train_raw, test_raw)
fecha_min = dates_train.min()
cat_idx   = [X_raw.columns.tolist().index(c) for c in CATEGORICAS if c in X_raw.columns]

# Pesos de clase (para XGBoost e LightGBM)
class_counts  = y_train.value_counts().sort_index()
class_weights = {c: len(y_train) / (len(class_counts) * n) for c, n in class_counts.items()}
sample_weight_full = y_train.map(class_weights).values
print(f"  Pesos de clase: { {k: round(v,3) for k,v in class_weights.items()} }")

# ── 3. Selección de variables ─────────────────────────────────────────────────
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

imp_cat = pd.Series(aux_model.get_feature_importance(aux_pool), index=X_raw.columns)

# Para OHE: mapear importancia das columnas OHE polo nome orixinal
imp_ohe = pd.Series(0.0, index=X_ohe.columns)
for col in X_ohe.columns:
    orig = next((c for c in X_raw.columns if col == c or col.startswith(c + '_')), col)
    imp_ohe[col] = imp_cat.get(orig, 0.0)

FEATURES_CAT = imp_cat.sort_values(ascending=False).index[:N_FEATURES].tolist()
FEATURES_OHE = imp_ohe.sort_values(ascending=False).index[:N_FEATURES].tolist()

print(f"\n  Top-{N_FEATURES} variables (CatBoost / sen OHE):")
for i, f in enumerate(FEATURES_CAT, 1):
    print(f"    {i:>2}. {f:<42}  imp: {imp_cat[f]:.4f}")

X_cat        = X_raw[FEATURES_CAT].copy()
T_cat        = T_raw[FEATURES_CAT].copy()
cat_idx_feat = [FEATURES_CAT.index(c) for c in CATEGORICAS if c in FEATURES_CAT]

X_xgb = X_ohe[FEATURES_OHE].copy()
T_xgb = T_ohe[FEATURES_OHE].copy()

# ── 4. Folds temporais ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
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
n_classes = y_train.nunique()
n_models  = len(CONFIGS_CAT) + len(CONFIGS_XGB) + len(CONFIGS_LGBM)

print("=" * 60)
print(f"  OOF STACKING  ({n_models} modelos: "
      f"{len(CONFIGS_CAT)} CatBoost + {len(CONFIGS_XGB)} XGBoost + {len(CONFIGS_LGBM)} LightGBM"
      f" × {len(folds)} folds)")
print("=" * 60)

oof_preds  = np.full((len(X_cat), n_models * n_classes), np.nan)
test_preds = np.zeros((n_models, len(folds), len(T_cat), n_classes))
f1_resumo  = []
m_global   = 0

# ── CatBoost ──────────────────────────────────────────────────────────────────
for cfg in CONFIGS_CAT:
    desc = f"iter={cfg['iterations']}, lr={cfg['learning_rate']}, depth={cfg['depth']}, seed={cfg['random_seed']}"
    print(f"\n  ── CatBoost  ({desc})")
    f1_folds = []
    for fold, (idx_tr, idx_val) in enumerate(folds, 1):
        X_tr, y_tr   = X_cat.iloc[idx_tr],  y_train.iloc[idx_tr]
        X_val, y_val = X_cat.iloc[idx_val], y_train.iloc[idx_val]

        m = CatBoostClassifier(**cfg, **PARAMS_CAT, early_stopping_rounds=50)
        m.fit(Pool(X_tr, label=y_tr, cat_features=cat_idx_feat),
              eval_set=Pool(X_val, label=y_val, cat_features=cat_idx_feat))

        proba = m.predict_proba(X_val)
        oof_preds[idx_val, m_global*n_classes:(m_global+1)*n_classes] = proba
        test_preds[m_global, fold-1] = m.predict_proba(Pool(T_cat, cat_features=cat_idx_feat))

        f1 = f1_score(y_val, proba.argmax(axis=1), average='macro')
        f1_folds.append(f1)
        print(f"    Fold {fold}  |  train ≤ {dates_train.iloc[idx_tr].max().date()}  |  "
              f"val ≥ {dates_train.iloc[idx_val].min().date()}  |  F1-macro = {f1:.5f}")

    f1_medio = float(np.mean(f1_folds))
    f1_resumo.append(('CatBoost', cfg['random_seed'], f1_medio, np.std(f1_folds)))
    print(f"    → F1-macro medio : {f1_medio:.5f}  (std: {np.std(f1_folds):.5f})")
    m_global += 1

# ── XGBoost ───────────────────────────────────────────────────────────────────
for cfg in CONFIGS_XGB:
    desc = f"n_est={cfg['n_estimators']}, lr={cfg['learning_rate']}, depth={cfg['max_depth']}, seed={cfg['random_state']}"
    print(f"\n  ── XGBoost  ({desc})")
    f1_folds = []

    for fold, (idx_tr, idx_val) in enumerate(folds, 1):
        X_tr, y_tr   = X_xgb.iloc[idx_tr],  y_train.iloc[idx_tr]
        X_val, y_val = X_xgb.iloc[idx_val], y_train.iloc[idx_val]
        w_tr         = sample_weight_full[idx_tr]

        m = XGBClassifier(
            **cfg,
            objective='multi:softprob',
            num_class=n_classes,
            eval_metric='mlogloss',
            verbosity=0,
            n_jobs=-1,
            early_stopping_rounds=50,
        )
        m.fit(X_tr, y_tr, sample_weight=w_tr,
              eval_set=[(X_val, y_val)], verbose=False)

        proba = m.predict_proba(X_val)
        oof_preds[idx_val, m_global*n_classes:(m_global+1)*n_classes] = proba
        test_preds[m_global, fold-1] = m.predict_proba(T_xgb)

        f1 = f1_score(y_val, proba.argmax(axis=1), average='macro')
        f1_folds.append(f1)
        print(f"    Fold {fold}  |  train ≤ {dates_train.iloc[idx_tr].max().date()}  |  "
              f"val ≥ {dates_train.iloc[idx_val].min().date()}  |  F1-macro = {f1:.5f}")

    f1_medio = float(np.mean(f1_folds))
    f1_resumo.append(('XGBoost', cfg['random_state'], f1_medio, np.std(f1_folds)))
    print(f"    → F1-macro medio : {f1_medio:.5f}  (std: {np.std(f1_folds):.5f})")
    m_global += 1

# ── LightGBM ──────────────────────────────────────────────────────────────────
for cfg in CONFIGS_LGBM:
    desc = f"n_est={cfg['n_estimators']}, lr={cfg['learning_rate']}, depth={cfg['max_depth']}, seed={cfg['random_state']}"
    print(f"\n  ── LightGBM  ({desc})")
    f1_folds = []

    for fold, (idx_tr, idx_val) in enumerate(folds, 1):
        X_tr, y_tr   = X_xgb.iloc[idx_tr],  y_train.iloc[idx_tr]
        X_val, y_val = X_xgb.iloc[idx_val], y_train.iloc[idx_val]
        w_tr         = sample_weight_full[idx_tr]

        m = LGBMClassifier(
            **cfg,
            objective='multiclass',
            num_class=n_classes,
            metric='multi_logloss',
            verbosity=-1,
            n_jobs=-1,
        )
        m.fit(X_tr, y_tr,
              sample_weight=w_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[])

        proba = m.predict_proba(X_val)
        oof_preds[idx_val, m_global*n_classes:(m_global+1)*n_classes] = proba
        test_preds[m_global, fold-1] = m.predict_proba(T_xgb)

        f1 = f1_score(y_val, proba.argmax(axis=1), average='macro')
        f1_folds.append(f1)
        print(f"    Fold {fold}  |  train ≤ {dates_train.iloc[idx_tr].max().date()}  |  "
              f"val ≥ {dates_train.iloc[idx_val].min().date()}  |  F1-macro = {f1:.5f}")

    f1_medio = float(np.mean(f1_folds))
    f1_resumo.append(('LightGBM', cfg['random_state'], f1_medio, np.std(f1_folds)))
    print(f"    → F1-macro medio : {f1_medio:.5f}  (std: {np.std(f1_folds):.5f})")
    m_global += 1

# ── Resumo ────────────────────────────────────────────────────────────────────
print(f"\n{'─' * 60}")
print("  RESUMO DE VALIDACIÓN:")
for modelo, seed, f1, std in f1_resumo:
    print(f"    {modelo:<10}  (seed={seed})  →  F1={f1:.5f}  std={std:.5f}")
print(f"    Media xeral : {np.mean([r[2] for r in f1_resumo]):.5f}")
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
    mean_test = np.zeros((len(T_cat), n_classes))
    for m in range(n_models):
        mean_test += test_preds[m].mean(axis=0)
    mean_test /= n_models
    final_preds = mean_test.argmax(axis=1).astype(int)
    metodo = 'media simple'

# ── 7. Exportar ───────────────────────────────────────────────────────────────
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