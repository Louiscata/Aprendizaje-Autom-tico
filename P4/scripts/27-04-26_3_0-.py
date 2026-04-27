import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

# ── 0. Configuración ─────────────────────────────────────────────────────────
SEED          = 777
CV_FOLDS      = 5
N_MIN, N_MAX  = 10, 20          # Rango de features a explorar
RUTA_TRAIN    = './data/train.csv'
RUTA_TEST     = './data/test.csv'

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

# 2a. Eliminar duplicados
n_antes = len(train)
train = train.drop_duplicates(keep='first').reset_index(drop=True)
print(f"  Duplicados eliminados : {n_antes - len(train)}")

# 2b. Separar target e IDs
y_train = train['Target_Risco'].copy()
ids_test = test['ID_Cliente'].copy()

COLS_DROP_TRAIN = ['ID_Cliente', 'Data_Solicitude', 'Target_Risco']
COLS_DROP_TEST  = ['ID_Cliente', 'Data_Solicitude']

X_train = train.drop(columns=[c for c in COLS_DROP_TRAIN if c in train.columns])
X_test  = test.drop(columns=[c for c in COLS_DROP_TEST  if c in test.columns])

# 2c. Identificar variables categóricas (CatBoost trátalas de forma nativa)
CAT_FEATURES = [
    col for col in X_train.columns
    if X_train[col].dtype == object or col in ('Codigo_Postal', 'Subscricion_Email',
                                                 'Historial_Impagos', 'Num_Fillos',
                                                 'Numero_Tarxetas', 'Prestamos_Activos',
                                                 'Consultas_Risco_6M')
]

# 2d. Imputar NaN: mediana en numéricas, 'DESCOÑECIDO' en categóricas
for col in X_train.columns:
    if col in CAT_FEATURES:
        fill = X_train[col].mode()[0] if X_train[col].notna().any() else 'DESCOÑECIDO'
        X_train[col] = X_train[col].fillna(fill).astype(str)
        X_test[col]  = X_test[col].fillna(fill).astype(str)
    else:
        median = X_train[col].median()
        X_train[col] = X_train[col].fillna(median)
        X_test[col]  = X_test[col].fillna(median)

# Reconvertir categóricas a str para CatBoost
for col in CAT_FEATURES:
    X_train[col] = X_train[col].astype(str)
    X_test[col]  = X_test[col].astype(str)

cat_idx_full = [X_train.columns.tolist().index(c) for c in CAT_FEATURES if c in X_train.columns]

print(f"  Variables totais       : {X_train.shape[1]}")
print(f"  Variables categóricas  : {len(CAT_FEATURES)} → {CAT_FEATURES}")
print(f"  Train final            : {X_train.shape}\n")

# ── 3. Importancia de variables (modelo auxiliar sobre todo o conxunto) ───────
print("=" * 60)
print("  RANKING DE VARIABLES (CatBoost importance)")
print("=" * 60)

aux_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    auto_class_weights='Balanced',
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

print("\n  Top-25 variables por importancia:")
for i, (feat, imp) in enumerate(importances.head(25).items(), 1):
    print(f"    {i:>2}. {feat:<35}  {imp:.4f}")
print()

RANKING = importances.index.tolist()  # lista ordenada de maior a menor importancia

# ── 4. Busca do mellor N (10 → 20) con validación cruzada ───────────────────
print("=" * 60)
print(f"  SELECCIÓN DE FEATURES  (N de {N_MIN} a {N_MAX}  ·  CV={CV_FOLDS})")
print("=" * 60)

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

resultados = {}   # {n: f1_medio}

for n in range(N_MIN, N_MAX + 1):
    features_n = RANKING[:n]
    cat_n = [c for c in CAT_FEATURES if c in features_n]
    cat_idx_n = [features_n.index(c) for c in cat_n]

    f1_folds = []
    for fold, (idx_tr, idx_val) in enumerate(skf.split(X_train, y_train), 1):
        X_tr  = X_train[features_n].iloc[idx_tr]
        y_tr  = y_train.iloc[idx_tr]
        X_val = X_train[features_n].iloc[idx_val]
        y_val = y_train.iloc[idx_val]

        pool_tr  = Pool(X_tr,  label=y_tr,  cat_features=cat_idx_n)
        pool_val = Pool(X_val, label=y_val, cat_features=cat_idx_n)

        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=7,
            auto_class_weights='Balanced',
            loss_function='MultiClass',
            eval_metric='TotalF1',
            random_seed=SEED,
            verbose=0,
            thread_count=-1,
            early_stopping_rounds=40,
        )
        model.fit(pool_tr, eval_set=pool_val)

        preds = model.predict(X_val).ravel().astype(int)
        f1 = f1_score(y_val, preds, average='macro')
        f1_folds.append(f1)

    f1_medio = float(np.mean(f1_folds))
    resultados[n] = f1_medio
    print(f"  N={n:>2}  →  F1-macro CV = {f1_medio:.5f}  "
          f"(folds: {' · '.join(f'{v:.4f}' for v in f1_folds)})")

# ── 5. Mellor N ─────────────────────────────────────────────────────────────
BEST_N = max(resultados, key=resultados.get)
BEST_F1 = resultados[BEST_N]
BEST_FEATURES = RANKING[:BEST_N]

print(f"\n{'─'*60}")
print(f"  ✔  Mellor N = {BEST_N}  →  F1-macro = {BEST_F1:.5f}")
print(f"  Features seleccionadas:")
for i, f in enumerate(BEST_FEATURES, 1):
    print(f"    {i:>2}. {f}")
print(f"{'─'*60}\n")

# ── 6. Adestramento final sobre todo o train co mellor N ────────────────────
print("=" * 60)
print("  ADESTRAMENTO FINAL")
print("=" * 60)

cat_best   = [c for c in CAT_FEATURES if c in BEST_FEATURES]
cat_idx_best = [BEST_FEATURES.index(c) for c in cat_best]

final_pool = Pool(
    X_train[BEST_FEATURES], label=y_train, cat_features=cat_idx_best
)

final_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=7,
    auto_class_weights='Balanced',
    loss_function='MultiClass',
    eval_metric='TotalF1',
    random_seed=SEED,
    verbose=100,
    thread_count=-1,
)
final_model.fit(final_pool)

# ── 7. Predición e arquivo de envío ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  PREDICIÓN E EXPORTACIÓN")
print("=" * 60)

pool_test = Pool(X_test[BEST_FEATURES], cat_features=cat_idx_best)
preds_test = final_model.predict(pool_test).ravel().astype(int)

submission = pd.DataFrame({
    'ID_Cliente':   ids_test,
    'Target_Risco': preds_test,
})
submission.to_csv(f"./resultados/26-04-2026_catboost_features{BEST_N}.csv", index=False)
print(f"\n  Arquivo xerado: '26-04-2026_catboost_features{BEST_N}.csv'  ({len(submission)} filas)")
print(f"  Distribución de clases preditas:\n{submission['Target_Risco'].value_counts().sort_index().to_string()}")
print("\n✔ Proceso completado con éxito.")