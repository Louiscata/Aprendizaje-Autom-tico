import pandas as pd
import numpy as np

from sklearn.ensemble import (
    ExtraTreesClassifier, 
    HistGradientBoostingClassifier, # [Mellora] Substitúe ao GradientBoostingClassifier
    RandomForestClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# LightGBM e XGBoost son os mellores modelos para datos tabulares en Kaggle.
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

# ── 0. Variables globais ──────────────────────────────────────────────────────
SEED          = 42
NUM_FEATURES  = 15    # Un bo número xa que MI xestiona ben a relevancia

# [FIX] Activamos as opcións que salvaron a clase 3
OUTLIERS        = 'capear' 
NORMALIZAR      = True
UMBRAL_NAN      = 10
TRANSFORMAR_LOG = True  

# ── 1. Cargar datos ───────────────────────────────────────────────────────────
train = pd.read_csv('./data/train.csv')
test  = pd.read_csv('./data/test.csv')

# ── 2. Preprocesamento completo ───────────────────────────────────────────────
X_train_full, X_test_full, y_train = preprocesar_datos(
    train,
    test,
    outliers=OUTLIERS,     # Lembra que no teu código puido chamarse diferente, axúastao se fai falla
    umbral_nan=UMBRAL_NAN,
    normalizar=NORMALIZAR,
    trans_log=TRANSFORMAR_LOG,  # Aplicar suavizado ás variables de diñeiro
)

# ── 3. Selección das mellores variables (Mutual Information) ──────────────────
features = seleccion_de_variables(X_train_full, y_train, NUM_FEATURES)

X_train = X_train_full[features].copy()
X_test  = X_test_full[features].copy()

# ── 4. Definición dos modelos base e do meta-modelo ──────────────────────────
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
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
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

# [FIX] Usamos HistGradientBoostingClassifier con class_weight='balanced'
modelos_base.append(('hist_gradient_boosting', HistGradientBoostingClassifier(
    max_iter=200,
    max_depth=4,
    learning_rate=0.05,
    class_weight='balanced',
    random_state=SEED,
)))

modelos_base.append(('svc', SVC(
    kernel='rbf',
    probability=True,
    class_weight='balanced',
    random_state=SEED,
    C=1.0,
)))

modelos_base.append(('mlp', MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=SEED,
)))

modelos_base.append(('knn', KNeighborsClassifier(
    n_neighbors=5,
    n_jobs=-1,
    weights='distance',
)))

# ── Meta-modelo ───────────────────────────────────────────────────────────────
# [FIX] Random Forest sinxelo como "xuíz" das probabilidades (sen class_weight)
meta_modelo = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=SEED
)

# ── 5. Adestramento por stacking ─────────────────────────────────────────────
print("\nIniciando Stacking (Paciencia, tardará un anaco por culpa do CV=5)...")
stacking_clf, preds = adestrar_por_stacking(
    modelos_base=modelos_base,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    meta_modelo=meta_modelo,
    cv=5,
    passthrough=False,    # [FIX] Falso, que só use as probabilidades!
    seed=SEED,
)

# ── 6. Arquivo de envío ───────────────────────────────────────────────────────
nome_arquivo = f'./resultados/22-04-2026_4_stacking_f{NUM_FEATURES}.csv'

submission = pd.DataFrame({
    'ID_Cliente':   test['ID_Cliente'],
    'Target_Risco': preds.astype(int), # O Stacking xa devolve enteiros
})
submission.to_csv(nome_arquivo, index=False)
print(f"Arquivo xerado: {nome_arquivo} ({len(submission)} filas)")