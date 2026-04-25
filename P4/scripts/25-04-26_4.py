import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Importamos os módulos V3
from funciones.preprocesadoV3 import preprocesar_datos, seleccion_de_variables
from funciones.modeladoV3 import adestrar_por_stacking

# Importamos os motores modernos
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("[AVISO] lightgbm non instalado.")

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[AVISO] xgboost non instalado.")

import warnings
warnings.filterwarnings('ignore')

# ── 0. Variables globais ─────────────────────────────────────────────────────
SEED = 777
NUM_FEATURES = 18

OUTLIERS        = None   
NORMALIZAR      = False  
UMBRAL_NAN      = 10
TRANSFORMAR_LOG = False  

# ── 1. Cargar datos ───────────────────────────────────────────────────────────
train = pd.read_csv('./data/train.csv')
test  = pd.read_csv('./data/test.csv')

# ── 2. Preprocesamento (Usa o teu módulo limpo) ───────────────────────────────
X_train_full, X_test_full, y_train = preprocesar_datos(
    train, 
    test, 
    outliers=OUTLIERS, 
    umbral_nan=UMBRAL_NAN,
    normalizar=NORMALIZAR, 
    trans_log=TRANSFORMAR_LOG,
    imputar_nan_test=False
)

# ── 3. Selección de variables ─────────────────────────────────────────────────
features = seleccion_de_variables(X_train_full, y_train, NUM_FEATURES)

X_train = X_train_full[features].copy()
X_test  = X_test_full[features].copy()

# Imputar nulos (mediana) só para o TOP de variables
for col in features:
    median = X_train[col].median()
    X_train[col] = X_train[col].fillna(median)
    X_test[col]  = X_test[col].fillna(median)

# ── 4. Definición dos modelos base ────────────────────────────────────────────
modelos_base = []

modelos_base.append(('random_forest', RandomForestClassifier(
    n_estimators=100, max_depth=10, class_weight='balanced', random_state=SEED, n_jobs=-1
)))

if LGBM_AVAILABLE:
    modelos_base.append(('lgbm', LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6, class_weight='balanced', 
        random_state=SEED, n_jobs=-1, verbose=-1
    )))

if XGB_AVAILABLE:
    modelos_base.append(('xgb', XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        eval_metric='mlogloss', random_state=SEED, n_jobs=-1, verbosity=0
    )))

meta_modelo = LogisticRegression(max_iter=1000, random_state=SEED)

# ── 5. Adestramento por stacking ──────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"  ADESTRANDO STACKING BASE - SEED {SEED}")
print("=" * 70 + "\n")

stacking_clf, preds = adestrar_por_stacking(
    modelos_base=modelos_base,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    meta_modelo=meta_modelo,
    cv=5,
    seed=SEED
)

# ── 6. Arquivo de envío ───────────────────────────────────────────────────────
nome_arquivo = f'./resultados/25-04-2026_4-f{NUM_FEATURES}.csv'

submission = pd.DataFrame({
    'ID_Cliente':   test['ID_Cliente'],
    'Target_Risco': preds.astype(int),
})
submission.to_csv(nome_arquivo, index=False)
print(f"Arquivo xerado con éxito: {nome_arquivo} ({len(submission)} filas)")