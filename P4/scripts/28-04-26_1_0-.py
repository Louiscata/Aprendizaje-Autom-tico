import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import VotingClassifier
from funciones.preprocesadoV3 import crear_features

import warnings
warnings.filterwarnings('ignore')

# ── 0. Configuración ─────────────────────────────────────────────────────────
SEED          = 31416
CV_FOLDS      = 5
N_MIN, N_MAX  = 12, 25  # Rango de variables a explorar (ampliado polas financeiras)

# ── 1. Carga e Enxeñaría de Variables ─────────────────────────────────────────
print("=" * 60)
print("  CARGA E ENXEÑARÍA DE VARIABLES FINANCEIRAS")
print("=" * 60)
train_raw = pd.read_csv('./data/train.csv')
test_raw  = pd.read_csv('./data/test.csv')

train = crear_features(train_raw)
test  = crear_features(test_raw)

train = train.drop_duplicates(keep='first').reset_index(drop=True)

y_train = train['Target_Risco'].copy()
X_train = train.drop(columns=['ID_Cliente', 'Data_Solicitude', 'Target_Risco'], errors='ignore')
X_test  = test.drop(columns=['ID_Cliente', 'Data_Solicitude'], errors='ignore')

# ── 2. Preparar Categorías Nativas para CatBoost ──────────────────────────────
# Protexemos o Código Postal para que sexa texto
for df in [X_train, X_test]:
    if 'Codigo_Postal' in df.columns:
        df['Codigo_Postal'] = 'CP_' + df['Codigo_Postal'].astype(str)

CAT_FEATURES = X_train.select_dtypes(include=['object']).columns.tolist()

for col in X_train.columns:
    if col in CAT_FEATURES:
        X_train[col] = X_train[col].fillna("Descoñecido").astype(str)
        X_test[col]  = X_test[col].fillna("Descoñecido").astype(str)
    else:
        median = X_train[col].median()
        X_train[col] = X_train[col].fillna(median)
        X_test[col]  = X_test[col].fillna(median)

print(f"  Total variables listas para competir: {X_train.shape[1]}")

# ── 3. Ranking de Variables (CatBoost Importance) ─────────────────────────────
print("\n" + "=" * 60)
print("  FASE 1: RANKING DE VARIABLES")
print("=" * 60)

aux_model = CatBoostClassifier(
    iterations=300, learning_rate=0.1, depth=6,
    auto_class_weights='Balanced', random_seed=SEED, verbose=0
)
# Pasámoslle as variables que son texto polo seu nome
aux_model.fit(X_train, y_train, cat_features=CAT_FEATURES)

importances = pd.Series(
    aux_model.get_feature_importance(),
    index=X_train.columns
).sort_values(ascending=False)

RANKING = importances.index.tolist()

# ── 4. Busca do Mellor N (Bucle de CV) ────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  FASE 2: BUSCANDO O 'N' PERFECTO (De {N_MIN} a {N_MAX})")
print("=" * 60)

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
resultados = {}

for n in range(N_MIN, N_MAX + 1):
    features_n = RANKING[:n]
    cat_n = [c for c in CAT_FEATURES if c in features_n]
    
    f1_folds = []
    for idx_tr, idx_val in skf.split(X_train, y_train):
        X_tr, y_tr   = X_train[features_n].iloc[idx_tr], y_train.iloc[idx_tr]
        X_val, y_val = X_train[features_n].iloc[idx_val], y_train.iloc[idx_val]

        model = CatBoostClassifier(
            iterations=400, learning_rate=0.05, depth=6,
            auto_class_weights='Balanced', random_seed=SEED, verbose=0,
            early_stopping_rounds=30
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=cat_n)
        
        preds = model.predict(X_val).flatten()
        f1_folds.append(f1_score(y_val, preds, average='macro'))

    f1_medio = np.mean(f1_folds)
    resultados[n] = f1_medio
    print(f"  N={n:>2}  →  CV F1-Macro = {f1_medio:.5f}")

BEST_N = max(resultados, key=resultados.get)
BEST_FEATURES = RANKING[:BEST_N]
FINAL_CAT_FEATURES = [c for c in CAT_FEATURES if c in BEST_FEATURES]

print(f"\n  Mellor configuración: Top {BEST_N} variables (F1: {resultados[BEST_N]:.5f})")

# ── 5. Ensamblaxe Final (O teu Soft Voting de CatBoosts) ──────────────────────
print("\n" + "=" * 60)
print("  FASE 3: ADESTRAMENTO DO COMITÉ (SOFT VOTING) CAS MELLORES VARIABLES")
print("=" * 60)

# Recortamos os datos á perfección descuberta
X_train_best = X_train[BEST_FEATURES]
X_test_best  = X_test[BEST_FEATURES]

cb_estandar = CatBoostClassifier(
    iterations=800, learning_rate=0.04, depth=6, l2_leaf_reg=3,
    auto_class_weights='Balanced', cat_features=FINAL_CAT_FEATURES,
    random_seed=42, verbose=0
)
cb_profundo = CatBoostClassifier(
    iterations=600, learning_rate=0.03, depth=8, l2_leaf_reg=5,
    auto_class_weights='Balanced', cat_features=FINAL_CAT_FEATURES,
    random_seed=31416, verbose=0
)
cb_superficial = CatBoostClassifier(
    iterations=1000, learning_rate=0.05, depth=4, l2_leaf_reg=1,
    auto_class_weights='Balanced', cat_features=FINAL_CAT_FEATURES,
    random_seed=777, verbose=0
)

ensamblaxe = VotingClassifier(
    estimators=[('est', cb_estandar), ('prof', cb_profundo), ('sup', cb_superficial)],
    voting='soft',
    n_jobs=-1
)

ensamblaxe.fit(X_train_best, y_train)

# ── 6. Predición e Exportación ────────────────────────────────────────────────
preds_test = ensamblaxe.predict(X_test_best).astype(int)

nome_arquivo = f'./resultados/27-04-2026-CAT_ENSEMBLE_TOP_{BEST_N}.csv'
submission = pd.DataFrame({
    'ID_Cliente': test_raw['ID_Cliente'],
    'Target_Risco': preds_test,
})
submission.to_csv(nome_arquivo, index=False)
print(f"\nArquivo xerado con éxito: {nome_arquivo}")