import pandas as pd
import numpy as np

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Importamos LightGBM e XGBoost
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("[AVISO] lightgbm non instalado. Executar: pip install lightgbm")

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[AVISO] xgboost non instalado. Executar: pip install xgboost")

from funciones.preprocesado import preprocesar_datos, seleccion_de_variables
from funciones.modelado import adestrar_por_stacking

import warnings
warnings.filterwarnings('ignore')

# ── 0. Variables globais ───────────────────────────────────
SEED            = 42
NUM_FEATURES    = 25
CV              = 5

OUTLIERS        = None
NORMALIZAR      = True
UMBRAL_NAN      = 10
TRANSFORMAR_LOG = False

# ── 1. Cargar datos ───────────────────────────────────────────────────────────
train = pd.read_csv('./data/train.csv')
test  = pd.read_csv('./data/test.csv')

# ── 2. Preprocesamento ────────────────────────────────────────────────────────
X_train_full, X_test_full, y_train = preprocesar_datos(
    train,
    test,
    outliers=OUTLIERS,
    umbral_nan=UMBRAL_NAN,
    normalizar=NORMALIZAR,
    trans_log=TRANSFORMAR_LOG,
)

# ── 3. Selección das mellores variables ────────────────────────────────────────
features = seleccion_de_variables(X_train_full, y_train, NUM_FEATURES)

X_train = X_train_full[features].copy()
X_test  = X_test_full[features].copy()

# ── 4. Definición dos modelos base  ────────────────────────────

modelos_base = []

if LGBM_AVAILABLE:
    modelos_base.append(('lgbm', LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )))

if XGB_AVAILABLE:
    modelos_base.append(('xgb', XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )))

modelos_base.append(('extra_trees', ExtraTreesClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight='balanced',
    min_samples_leaf=2,
    random_state=SEED,
    n_jobs=-1,
)))

modelos_base.append(('grad_boost', GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=SEED,
)))

modelos_base.append(('knn', KNeighborsClassifier(
    n_neighbors=5,
    n_jobs=-1,
    weights='distance',
)))

# ── Meta-modelo ───────────────────────────────────────────────────────────────
meta_modelo = LogisticRegression(
    max_iter=1000, 
    C=0.1, 
    random_state=SEED,
)

# ── 5. Adestramento por stacking ──────────────────────────────────────────────

stacking_clf, preds = adestrar_por_stacking(
    modelos_base=modelos_base,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    meta_modelo=meta_modelo,
    cv=CV,
    passthrough=False,
    seed=SEED,
)

# ── 6. Arquivo de envío ───────────────────────────────────────────────────────
nome_arquivo = f'./resultados/23-04-2026_2_stacking-f{NUM_FEATURES}.csv'

submission = pd.DataFrame({
    'ID_Cliente':   test['ID_Cliente'],
    'Target_Risco': preds.astype(int),
})
submission.to_csv(nome_arquivo, index=False)
print(f"\nArquivo xerado: {nome_arquivo} ({len(submission)} filas)")