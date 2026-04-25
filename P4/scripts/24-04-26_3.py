import pandas as pd
import numpy as np

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import f1_score
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

# ── 0. Variables globais  ────────────────────────────────────────────────────
SEED            = 42
NUM_FEATURES    = 20     # Co modelo cru, 20 variables íanche xenial

# A BASE DO TEU RÉCORD: DATOS CRUS! Ás árbores encántalles a desorde.
OUTLIERS        = None
NORMALIZAR      = False  # Sen KNN/MLP non fai falta
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

# ── 3. Selección das mellores variables  ───────────────────────────────────────
features = seleccion_de_variables(X_train_full, y_train, NUM_FEATURES)

X_train = X_train_full[features].copy()
X_test  = X_test_full[features].copy()

# Confirmar que non teñan NaN (O salvavidas)
for col in features:
    mediana_train = X_train[col].median()
    X_train[col] = X_train[col].fillna(mediana_train)
    X_test[col]  = X_test[col].fillna(mediana_train)

# ── 4. Definición dos modelos base (SÓ ÁRBORES) ───────────────────────────────
modelos_base = []

# O teu comité salvaxe
modelos_base.append(('random_forest', RandomForestClassifier(
    n_estimators=100, max_depth=10, class_weight='balanced', random_state=SEED, n_jobs=-1
)))

if LGBM_AVAILABLE:
    modelos_base.append(('lgbm', LGBMClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=6, class_weight='balanced',
        random_state=SEED, n_jobs=-1, verbose=-1
    )))

if XGB_AVAILABLE:
    modelos_base.append(('xgb', XGBClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=5,
        eval_metric='mlogloss', random_state=SEED, n_jobs=-1, verbosity=0
    )))

modelos_base.append(('extra_trees', ExtraTreesClassifier(
    n_estimators=200, max_depth=10, class_weight='balanced',
    random_state=SEED, n_jobs=-1
)))

# ── Meta-modelo ───────────────────────────────────────────────────────────────
meta_modelo = LogisticRegression(
    max_iter=1000,
    C=0.1, 
    random_state=SEED
)

# ── 5. Adestramento por stacking ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("  O MODELO DO RÉCORD (DATOS CRUS) + THRESHOLDING")
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

# ── 6. OTIMIZACIÓN DE LIMIARES (A Táctica dos Campións) ──────────────────────
print("\nBuscando o limiar óptimo para a Clase 3...")

probabilidades_train = stacking_clf.predict_proba(X_train)
probabilidades_test  = stacking_clf.predict_proba(X_test)

best_f1 = 0
best_thresh = 0.25 

for thresh in np.linspace(0.10, 0.40, 60):
    preds_temporais = np.argmax(probabilidades_train, axis=1)
    preds_temporais[probabilidades_train[:, 3] > thresh] = 3
    
    score = f1_score(y_train, preds_temporais, average='macro')
    if score > best_f1:
        best_f1 = score
        best_thresh = thresh

print(f"Limiar óptimo atopado para Clase 3: {best_thresh:.4f}")
print(f"F1-Macro no Train despois do Thresholding: {best_f1:.4f}")

# ── 7. Aplicar a maxia ao Test e gardar ──────────────────────────────────────
preds_test_opt = np.argmax(probabilidades_test, axis=1)
preds_test_opt[probabilidades_test[:, 3] > best_thresh] = 3

nome_arquivo = f'./resultados/24-04-2026-3_f{NUM_FEATURES}.csv'

submission = pd.DataFrame({
    'ID_Cliente':   test['ID_Cliente'],
    'Target_Risco': preds_test_opt.astype(int),
})
submission.to_csv(nome_arquivo, index=False)
print(f"Arquivo de subida xerado con éxito: {nome_arquivo} ({len(submission)} filas)")