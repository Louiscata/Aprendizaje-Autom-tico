"""
catboost_sum_models.py
──────────────────────
Competición Kaggle – Clasificación de risco crediticio (Target_Risco: 0-3)

Estratexia:
  1. Preprocesado mínimo: eliminación de duplicados + imputación básica
  2. Selección de variables: top-17 por importancia de CatBoost (modelo auxiliar)
  3. Adestramento de N modelos CatBoost con hiperparámetros variados
  4. Combinación con catboost.sum_models() — promedia as follas dos árbores
  5. Predición e exportación
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool, sum_models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

# ── 0. Configuración ─────────────────────────────────────────────────────────
SEED        = 777
CV_FOLDS    = 5
N_FEATURES  = 17
RUTA_TRAIN  = './data/train.csv'
RUTA_TEST   = './data/test.csv'

# ── 1. Carga de datos ────────────────────────────────────────────────────────
print("=" * 60)
print("  CARGA DE DATOS")
print("=" * 60)
train = pd.read_csv(RUTA_TRAIN)
test  = pd.read_csv(RUTA_TEST)
print(f"  Train bruto : {train.shape[0]} filas · {train.shape[1]} columnas")
print(f"  Test  bruto : {test.shape[0]} filas · {test.shape[1]} columnas\n")

# ── 2. Preprocesado mínimo ───────────────────────────────────────────────────
print("=" * 60)
print("  PREPROCESADO")
print("=" * 60)

n_antes = len(train)
train = train.drop_duplicates(keep='first').reset_index(drop=True)
print(f"  Duplicados eliminados : {n_antes - len(train)}")

y_train  = train['Target_Risco'].copy()
ids_test = test['ID_Cliente'].copy()

COLS_DROP_TRAIN = ['ID_Cliente', 'Data_Solicitude', 'Target_Risco']
COLS_DROP_TEST  = ['ID_Cliente', 'Data_Solicitude']

X_train = train.drop(columns=[c for c in COLS_DROP_TRAIN if c in train.columns])
X_test  = test.drop(columns=[c for c in COLS_DROP_TEST  if c in test.columns])

CAT_FEATURES = [
    col for col in X_train.columns
    if X_train[col].dtype == object or col in (
        'Codigo_Postal', 'Subscricion_Email', 'Historial_Impagos',
        'Num_Fillos', 'Numero_Tarxetas', 'Prestamos_Activos', 'Consultas_Risco_6M'
    )
]

for col in X_train.columns:
    if col in CAT_FEATURES:
        fill = X_train[col].mode()[0] if X_train[col].notna().any() else 'DESCOÑECIDO'
        X_train[col] = X_train[col].fillna(fill).astype(str)
        X_test[col]  = X_test[col].fillna(fill).astype(str)
    else:
        median = X_train[col].median()
        X_train[col] = X_train[col].fillna(median)
        X_test[col]  = X_test[col].fillna(median)

cat_idx_full = [X_train.columns.tolist().index(c) for c in CAT_FEATURES if c in X_train.columns]

print(f"  Variables totais      : {X_train.shape[1]}")
print(f"  Variables categóricas : {len(CAT_FEATURES)}")
print(f"  Train final           : {X_train.shape}\n")

# ── 3. Selección de top-17 variables por importancia ────────────────────────
print("=" * 60)
print(f"  SELECCIÓN DE TOP-{N_FEATURES} VARIABLES (CatBoost importance)")
print("=" * 60)

aux_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass',
    eval_metric='TotalF1',
    random_seed=SEED,
    verbose=0,
    thread_count=-1,
)
aux_pool = Pool(X_train, label=y_train, cat_features=cat_idx_full)
aux_model.fit(aux_pool)

importances = pd.Series(
    aux_model.get_feature_importance(aux_pool),
    index=X_train.columns
).sort_values(ascending=False)

FEATURES = importances.index[:N_FEATURES].tolist()

print(f"\n  Top-{N_FEATURES} variables seleccionadas:")
for i, feat in enumerate(FEATURES, 1):
    print(f"    {i:>2}. {feat:<35}  (imp: {importances[feat]:.4f})")
print()

# Reducir X e índices de categóricas ó subconxunto final
X_train_sel = X_train[FEATURES].copy()
X_test_sel  = X_test[FEATURES].copy()

cat_sel      = [c for c in CAT_FEATURES if c in FEATURES]
cat_idx_sel  = [FEATURES.index(c) for c in cat_sel]

pool_full_train = Pool(X_train_sel, label=y_train, cat_features=cat_idx_sel)
pool_test       = Pool(X_test_sel,  cat_features=cat_idx_sel)

# ── 4. Definición dos modelos a combinar ─────────────────────────────────────
# sum_models() promedia as estruturas internas dos árbores: precisa que todos
# teñan o mesmo loss_function, eval_metric e cat_features.
# Variamos: depth, learning_rate, regularización (l2, bagging_temperature,
# random_strength) e seed para maximizar diversidade.
print("=" * 60)
print("  DEFINICIÓN DO ENSEMBLE  (6 CatBoost + sum_models)")
print("=" * 60)

CONFIGS = [
    # (nome,            depth, lr,    iterations, l2,   bag_temp, rand_str, seed)
    ("base_profundo",      8,  0.03,    1200,      3.0,   1.0,     1.0,    SEED),
    ("rapido_shallow",     5,  0.07,     800,      1.0,   0.5,     0.5,    SEED + 1),
    ("regularizado",       7,  0.03,    1200,      8.0,   1.5,     2.0,    SEED + 2),
    ("alto_bagging",       6,  0.05,    1000,      2.0,   5.0,     1.0,    SEED + 3),
    ("deep_slow",          9,  0.02,    1500,      4.0,   1.0,     0.5,    SEED + 4),
    ("medio_diverso",      7,  0.04,    1000,      2.0,   2.0,     3.0,    SEED + 5),
]

# ── 5. Validación cruzada de cada modelo e do ensemble ───────────────────────
print("\n  Validación cruzada individual por modelo (CV-5, F1 macro):\n")

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

f1_por_modelo = {cfg[0]: [] for cfg in CONFIGS}
f1_ensemble_folds = []

for fold, (idx_tr, idx_val) in enumerate(skf.split(X_train_sel, y_train), 1):
    X_tr  = X_train_sel.iloc[idx_tr]
    y_tr  = y_train.iloc[idx_tr]
    X_val = X_train_sel.iloc[idx_val]
    y_val = y_train.iloc[idx_val]

    pool_tr  = Pool(X_tr,  label=y_tr,  cat_features=cat_idx_sel)
    pool_val = Pool(X_val, label=y_val, cat_features=cat_idx_sel)

    modelos_fold = []
    for nome, depth, lr, iters, l2, bag_temp, rand_str, seed in CONFIGS:
        m = CatBoostClassifier(
            iterations=iters,
            learning_rate=lr,
            depth=depth,
            loss_function='MultiClass',
            eval_metric='TotalF1',
            l2_leaf_reg=l2,
            bagging_temperature=bag_temp,
            random_strength=rand_str,
            random_seed=seed,
            early_stopping_rounds=50,
            verbose=0,
            thread_count=-1,
        )
        m.fit(pool_tr, eval_set=pool_val)

        preds_m = m.predict(X_val).ravel().astype(int)
        f1_m = f1_score(y_val, preds_m, average='macro')
        f1_por_modelo[nome].append(f1_m)
        modelos_fold.append(m)

    # Ensemble do fold con pesos iguais
    ensemble_fold = sum_models(
        modelos_fold,
        weights=[1 / len(modelos_fold)] * len(modelos_fold)
    )
    preds_ens = ensemble_fold.predict(X_val).ravel().astype(int)
    f1_ens = f1_score(y_val, preds_ens, average='macro')
    f1_ensemble_folds.append(f1_ens)

    print(f"  Fold {fold}:")
    for nome, _, _, _, _, _, _, _ in CONFIGS:
        print(f"    · {nome:<22}  F1={f1_por_modelo[nome][-1]:.5f}")
    print(f"    · {'ENSEMBLE':<22}  F1={f1_ens:.5f}")
    print()

print("  Resumo CV (media ± std):")
for nome, _, _, _, _, _, _, _ in CONFIGS:
    vals = f1_por_modelo[nome]
    print(f"    · {nome:<22}  {np.mean(vals):.5f} ± {np.std(vals):.5f}")
print(f"    · {'ENSEMBLE':<22}  {np.mean(f1_ensemble_folds):.5f} ± {np.std(f1_ensemble_folds):.5f}")
print()

# ── 6. Adestramento final de todos os modelos sobre o train completo ──────────
print("=" * 60)
print("  ADESTRAMENTO FINAL (train completo)")
print("=" * 60)

modelos_finais = []
for nome, depth, lr, iters, l2, bag_temp, rand_str, seed in CONFIGS:
    print(f"  Adestando: {nome} ...")
    m = CatBoostClassifier(
        iterations=iters,
        learning_rate=lr,
        depth=depth,
        loss_function='MultiClass',
        eval_metric='TotalF1',
        l2_leaf_reg=l2,
        bagging_temperature=bag_temp,
        random_strength=rand_str,
        random_seed=seed,
        verbose=0,
        thread_count=-1,
    )
    m.fit(pool_full_train)
    modelos_finais.append(m)
    print(f"    ✔ {nome} completado.")

# ── 7. Combinación con sum_models ────────────────────────────────────────────
print(f"\n  Combinando {len(modelos_finais)} modelos con sum_models (pesos iguais)...")
ensemble_final = sum_models(
    modelos_finais,
    weights=[1 / len(modelos_finais)] * len(modelos_finais)
)
print("  ✔ Ensemble creado.\n")

# ── 8. Predición e arquivo de envío ─────────────────────────────────────────
print("=" * 60)
print("  PREDICIÓN E EXPORTACIÓN")
print("=" * 60)

preds_test = ensemble_final.predict(pool_test).ravel().astype(int)

nome_arquivo = f'./resultados/27-04-2026_catboost_sum{len(CONFIGS)}_features{N_FEATURES}.csv'
submission = pd.DataFrame({
    'ID_Cliente':   ids_test,
    'Target_Risco': preds_test,
})
submission.to_csv(nome_arquivo, index=False)

print(f"\n  Arquivo xerado : {nome_arquivo}  ({len(submission)} filas)")
print(f"  F1-macro CV ensemble : {np.mean(f1_ensemble_folds):.5f} ± {np.std(f1_ensemble_folds):.5f}")
print(f"  Distribución de clases preditas:")
print(submission['Target_Risco'].value_counts().sort_index().to_string())
print("\n✔ Proceso completado con éxito.")