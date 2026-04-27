import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# ── 0. Configuración ─────────────────────────────────────────────────────────
SEED       = 777
CV_FOLDS   = 5
TOP_N      = 17          # Mellor N segundo os experimentos do compañeiro
RUTA_TRAIN = './data/train.csv'
RUTA_TEST  = './data/test.csv'
RUTA_OUT   = './27-04-2026_4_catboost_stacking.csv'

# ── 1. Carga de datos ─────────────────────────────────────────────────────────
print("=" * 60)
print("  CARGA DE DATOS")
print("=" * 60)
train = pd.read_csv(RUTA_TRAIN)
test  = pd.read_csv(RUTA_TEST)
print(f"  Train: {train.shape}  |  Test: {test.shape}")

# ── 2. Preprocesado ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PREPROCESADO")
print("=" * 60)

train = train.drop_duplicates(keep='first').reset_index(drop=True)

y     = train['Target_Risco'].copy()
ids_test = test['ID_Cliente'].copy()

DROP_TRAIN = ['ID_Cliente', 'Data_Solicitude', 'Target_Risco']
DROP_TEST  = ['ID_Cliente', 'Data_Solicitude']

X_all  = train.drop(columns=[c for c in DROP_TRAIN if c in train.columns])
X_test = test.drop(columns=[c for c in DROP_TEST  if c in test.columns])

CAT_FEATURES = [
    col for col in X_all.columns
    if X_all[col].dtype == object or col in (
        'Codigo_Postal', 'Subscricion_Email', 'Historial_Impagos',
        'Num_Fillos', 'Numero_Tarxetas', 'Prestamos_Activos', 'Consultas_Risco_6M'
    )
]

for col in X_all.columns:
    if col in CAT_FEATURES:
        fill = X_all[col].mode()[0] if X_all[col].notna().any() else 'DESCOÑECIDO'
        X_all[col]  = X_all[col].fillna(fill).astype(str)
        X_test[col] = X_test[col].fillna(fill).astype(str)
    else:
        median = X_all[col].median()
        X_all[col]  = X_all[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)

cat_idx_full = [X_all.columns.tolist().index(c) for c in CAT_FEATURES if c in X_all.columns]
print(f"  Categóricas: {CAT_FEATURES}")
print(f"  Shape final: {X_all.shape}")

# ── 3. Ranking de variables (modelo auxiliar) ─────────────────────────────────
print("\n" + "=" * 60)
print("  RANKING DE VARIABLES")
print("=" * 60)

aux = CatBoostClassifier(
    iterations=300, learning_rate=0.1, depth=6,
    loss_function='MultiClass', eval_metric='TotalF1',
    random_seed=SEED, verbose=0, thread_count=-1,
)
aux.fit(Pool(X_all, label=y, cat_features=cat_idx_full))

importances = pd.Series(
    aux.get_feature_importance(Pool(X_all, label=y, cat_features=cat_idx_full)),
    index=X_all.columns
).sort_values(ascending=False)

RANKING       = importances.index.tolist()
BEST_FEATURES = RANKING[:TOP_N]

print(f"\n  Top-{TOP_N} features seleccionadas:")
for i, f in enumerate(BEST_FEATURES, 1):
    print(f"    {i:>2}. {f:<40} {importances[f]:.4f}")

cat_best    = [c for c in CAT_FEATURES if c in BEST_FEATURES]
cat_idx_best = [BEST_FEATURES.index(c) for c in cat_best]

X = X_all[BEST_FEATURES].copy()
T = X_test[BEST_FEATURES].copy()

# ── 4. Configuración dos modelos base ─────────────────────────────────────────
# Tres CatBoosts con hiperparámetros distintos para maximizar diversidade
BASE_CONFIGS = [
    dict(  # Modelo A – conservador, árbores profundas
        iterations=800, learning_rate=0.03, depth=8,
        loss_function='MultiClass', eval_metric='TotalF1',
        random_seed=SEED, verbose=0, thread_count=-1,
        early_stopping_rounds=50,
        l2_leaf_reg=5, bagging_temperature=0.5,
    ),
    dict(  # Modelo B – rápido, regularizado
        iterations=600, learning_rate=0.06, depth=6,
        loss_function='MultiClass', eval_metric='TotalF1',
        random_seed=SEED + 1, verbose=0, thread_count=-1,
        early_stopping_rounds=50,
        l2_leaf_reg=10, subsample=0.8, colsample_bylevel=0.8,
        bootstrap_type='Bernoulli',
    ),
    dict(  # Modelo C – máis iteracións, leaf regularization alta
        iterations=1000, learning_rate=0.02, depth=7,
        loss_function='MultiClass', eval_metric='TotalF1',
        random_seed=SEED + 2, verbose=0, thread_count=-1,
        early_stopping_rounds=60,
        l2_leaf_reg=15, min_data_in_leaf=20,
    ),
]

n_classes = y.nunique()   # 4
n_models  = len(BASE_CONFIGS)

# ── 5. Out-Of-Fold para o meta-learner ────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  OOF STACKING  ({n_models} modelos base × {CV_FOLDS} folds)")
print("=" * 60)

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

# Matrices OOF: (n_train, n_models * n_classes)
oof_preds  = np.zeros((len(X), n_models * n_classes))
# Predicciones test: (n_models, CV_FOLDS, n_test, n_classes)
test_preds = np.zeros((n_models, CV_FOLDS, len(T), n_classes))

for m_idx, cfg in enumerate(BASE_CONFIGS):
    print(f"\n  ── Modelo {m_idx + 1}/{n_models} ──")
    for fold, (idx_tr, idx_val) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr   = X.iloc[idx_tr],  y.iloc[idx_tr]
        X_val, y_val = X.iloc[idx_val], y.iloc[idx_val]

        pool_tr  = Pool(X_tr,  label=y_tr,  cat_features=cat_idx_best)
        pool_val = Pool(X_val, label=y_val, cat_features=cat_idx_best)

        model = CatBoostClassifier(**cfg)
        model.fit(pool_tr, eval_set=pool_val)

        # Probabilidades OOF
        proba_val = model.predict_proba(X_val)
        oof_preds[idx_val, m_idx * n_classes:(m_idx + 1) * n_classes] = proba_val

        # Probabilidades test (promedio dos folds despois)
        pool_test = Pool(T, cat_features=cat_idx_best)
        test_preds[m_idx, fold - 1] = model.predict_proba(pool_test)

        # F1 do fold
        preds_hard = proba_val.argmax(axis=1)
        f1 = f1_score(y_val, preds_hard, average='macro')
        print(f"    Fold {fold}  →  F1-macro = {f1:.5f}")

# Promedio das prediccións test sobre os folds
# Shape final test meta: (n_test, n_models * n_classes)
test_meta = np.hstack([
    test_preds[m].mean(axis=0) for m in range(n_models)
])

# F1 OOF global (usando argmax das probabilidades medias de todos os modelos)
mean_oof_proba = np.zeros((len(X), n_classes))
for m in range(n_models):
    mean_oof_proba += oof_preds[:, m * n_classes:(m + 1) * n_classes]
mean_oof_proba /= n_models
f1_oof_ensemble = f1_score(y, mean_oof_proba.argmax(axis=1), average='macro')
print(f"\n  F1-macro OOF (media simple dos modelos base): {f1_oof_ensemble:.5f}")

# ── 6. Meta-learner (Regresión Loxística) ─────────────────────────────────────
print("\n" + "=" * 60)
print("  META-LEARNER (Logistic Regression)")
print("=" * 60)

meta = LogisticRegression(
    C=1.0, max_iter=1000, random_state=SEED,
    multi_class='multinomial', solver='lbfgs',
)
meta.fit(oof_preds, y)

# F1 do meta-learner sobre OOF
meta_oof_preds = meta.predict(oof_preds)
f1_meta = f1_score(y, meta_oof_preds, average='macro')
print(f"  F1-macro OOF meta-learner: {f1_meta:.5f}")

# ── 7. Predicción final ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PREDICCIÓN FINAL")
print("=" * 60)

final_preds = meta.predict(test_meta).astype(int)

submission = pd.DataFrame({
    'ID_Cliente':   ids_test,
    'Target_Risco': final_preds,
})
submission.to_csv(RUTA_OUT, index=False)

print(f"  Arquivo: '{RUTA_OUT}'  ({len(submission)} filas)")
print(f"  Distribución de clases:\n{submission['Target_Risco'].value_counts().sort_index().to_string()}")
print("\n✔ Proceso completado con éxito.")