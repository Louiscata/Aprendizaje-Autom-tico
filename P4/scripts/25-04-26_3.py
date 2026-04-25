import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from funciones.preprocesadoV3 import preprocesar_datos, seleccion_de_variables

import warnings
warnings.filterwarnings('ignore')

# ── 0. Variables globais ─────────────────────────────────────────────────────
SEED = 42
NUM_FEATURES = 25

OUTLIERS        = None   
NORMALIZAR      = False  
UMBRAL_NAN      = 10
TRANSFORMAR_LOG = False  

# ── 1. Cargar datos ───────────────────────────────────────────────────────────
train = pd.read_csv('./data/train.csv')
test  = pd.read_csv('./data/test.csv')

# ── 2. Preprocesamento (Cos ratios financeiros activos) ───────────────────────
X_train_full, X_test_full, y_train = preprocesar_datos(
    train, test, outliers=OUTLIERS, umbral_nan=UMBRAL_NAN,
    normalizar=NORMALIZAR, trans_log=TRANSFORMAR_LOG,
)

# ── 3. Selección das mellores variables (Pearson) ─────────────────────────────
features = seleccion_de_variables(X_train_full, y_train, NUM_FEATURES)

X_train = X_train_full[features].copy()
X_test  = X_test_full[features].copy()

for col in features:
    mediana_train = X_train[col].median()
    X_train[col] = X_train[col].fillna(mediana_train)
    X_test[col]  = X_test[col].fillna(mediana_train)

# ── 4. Definición dos modelos base (Motores Optuna) ───────────────────────────
modelos_base = []

# 1. Random Forest (Apoio - Peso: 1)
modelos_base.append(('rf', RandomForestClassifier(
    n_estimators=100, max_depth=10, class_weight='balanced', random_state=SEED, n_jobs=-1
)))

# 2. LightGBM (Estrela - Peso: 4)
if LGBM_AVAILABLE:
    modelos_base.append(('lgbm', LGBMClassifier(
        n_estimators=456, learning_rate=0.0269, max_depth=10, num_leaves=65,
        subsample=0.6379, colsample_bytree=0.6767, min_child_samples=14,
        class_weight='balanced', random_state=SEED, n_jobs=-1, verbose=-1
    )))

# 3. XGBoost (Estrela - Peso: 4)
if XGB_AVAILABLE:
    modelos_base.append(('xgb', XGBClassifier(
        n_estimators=249, learning_rate=0.0358, max_depth=9,
        subsample=0.7960, colsample_bytree=0.8474, gamma=1.6974, min_child_weight=1,
        eval_metric='mlogloss', random_state=SEED, n_jobs=-1, verbosity=0
    )))

# 4. ExtraTrees (Apoio - Peso: 1)
modelos_base.append(('et', ExtraTreesClassifier(
    n_estimators=200, max_depth=10, class_weight='balanced', random_state=SEED, n_jobs=-1
)))

# ── 5. A Maxia: Weighted Voting (Sen sobreaxuste) ─────────────────────────────
# Pesos: 10% RF, 40% LGBM, 40% XGBoost, 10% ExtraTrees
pesos_modelos = [1, 4, 4, 1] 

ensamblaxe = VotingClassifier(
    estimators=modelos_base,
    voting='soft',
    weights=pesos_modelos,
    n_jobs=-1
)

# ── 6. Adestramento directo e Envío ───────────────────────────────────────────
print("\n" + "=" * 70)
print("  ADESTRANDO WEIGHTED ENSEMBLE")
print("=" * 70 + "\n")

# Para ver como se comporta en local antes de xerar
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(ensamblaxe, X_train, y_train, cv=skf, scoring='f1_macro')
print(f"  CV F1-Macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

# Adestramento final
ensamblaxe.fit(X_train, y_train)

nome_arquivo = f'./resultados/25-04-2026-ponderado_f{NUM_FEATURES}.csv'
preds_test = ensamblaxe.predict(X_test)

submission = pd.DataFrame({
    'ID_Cliente':   test['ID_Cliente'],
    'Target_Risco': preds_test.astype(int),
})
submission.to_csv(nome_arquivo, index=False)
print(f"Arquivo xerado con éxito: {nome_arquivo}")