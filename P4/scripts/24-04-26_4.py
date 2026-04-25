import pandas as pd
import numpy as np

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression

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

from funciones.preprocesadoV3 import preprocesar_datos, seleccion_de_variables
from funciones.modeladoV3 import adestrar_por_stacking

import warnings
warnings.filterwarnings('ignore')

# ── 0. Variables globais (A BASE DO RÉCORD) ──────────────────────────────────
SEED            = 42
NUM_FEATURES    = 25

# DATOS CRUS: O segredo para que XGBoost e LightGBM devoren a información
OUTLIERS        = None   
NORMALIZAR      = False  
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

# ── 3. Selección das mellores variables (PEARSON) ─────────────────────────────
features = seleccion_de_variables(X_train_full, y_train, NUM_FEATURES)

X_train = X_train_full[features].copy()
X_test  = X_test_full[features].copy()

# Confirmar que non teñan NaN (O salvavidas por se as derivadas fallan)
for col in features:
    mediana_train = X_train[col].median()
    X_train[col] = X_train[col].fillna(mediana_train)
    X_test[col]  = X_test[col].fillna(mediana_train)

# ── 4. Definición dos modelos base (OS MOTORES OPTIMIZADOS) ───────────────────
modelos_base = []

# 1. Random Forest Clásico
modelos_base.append(('random_forest', RandomForestClassifier(
    n_estimators=100, max_depth=10, class_weight='balanced', random_state=SEED, n_jobs=-1
)))

# 2. LightGBM (Afinado por Optuna)
if LGBM_AVAILABLE:
    modelos_base.append(('lgbm', LGBMClassifier(
        n_estimators=456,
        learning_rate=0.02692949001193457,
        max_depth=10,
        num_leaves=65,
        subsample=0.6379269967240289,
        colsample_bytree=0.6767132981034846,
        min_child_samples=14,
        class_weight='balanced', # Crucial para evitar o thresholding manual
        random_state=SEED, 
        n_jobs=-1, 
        verbose=-1
    )))

# 3. XGBoost (Afinado por Optuna)
if XGB_AVAILABLE:
    modelos_base.append(('xgb', XGBClassifier(
        n_estimators=249,
        learning_rate=0.035832074939857926,
        max_depth=9,
        subsample=0.7960001402726308,
        colsample_bytree=0.8474527760087138,
        gamma=1.6974523345977086,
        min_child_weight=1,
        eval_metric='mlogloss', 
        random_state=SEED, 
        n_jobs=-1, 
        verbosity=0
    )))

# 4. ExtraTrees (Achega diversidade e combate o sobreaxuste)
modelos_base.append(('extra_trees', ExtraTreesClassifier(
    n_estimators=200, max_depth=10, class_weight='balanced',
    random_state=SEED, n_jobs=-1
)))

# ── Meta-modelo ───────────────────────────────────────────────────────────────
# O Xuíz Neutral: Deixa as decisións de pesos ás árbores
meta_modelo = LogisticRegression(
    max_iter=1000,
    random_state=SEED
)

# ── 5. Adestramento por stacking ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("  O MODELO DO RÉCORD CON MOTORES OPTUNA")
print(f"  FEATURES={NUM_FEATURES}, OUTLIERS={OUTLIERS}, LOG={TRANSFORMAR_LOG}, NORM={NORMALIZAR}")
print("=" * 70 + "\n")

stacking_clf, preds = adestrar_por_stacking(
    modelos_base=modelos_base,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    meta_modelo=meta_modelo,
    cv=5,
    seed=SEED,
)

# ── 6. Arquivo de envío ───────────────────────────────────────────────────────
# Usamos a predición directa, sen thresholding, confiando na harmonía do código
nome_arquivo = f'./resultados/24-04-2026-4_f{NUM_FEATURES}.csv'

submission = pd.DataFrame({
    'ID_Cliente':   test['ID_Cliente'],
    'Target_Risco': preds.astype(int),
})
submission.to_csv(nome_arquivo, index=False)
print(f"Arquivo de subida xerado con éxito: {nome_arquivo} ({len(submission)} filas)")