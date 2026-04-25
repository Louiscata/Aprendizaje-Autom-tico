import pandas as pd
import numpy as np

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
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
NUM_FEATURES    = 18     # O noso punto doce matemático (nin underfitting nin ruído)
CV              = 5      # O StratifiedKFold xa está implementado no modeladoV3

# O preprocesado que protexe ás matemáticas pero mantén a información
OUTLIERS        = 'capear' # Domamos aos millonarios para o MLP e a Loxística
NORMALIZAR      = True     # Imprescindible para que o KNN non morra
UMBRAL_NAN      = 10
TRANSFORMAR_LOG = True     # Comprime as colas do diñeiro e suaviza os ratios

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

# Confirmar que non teñan NaN
for col in features:
    mediana_train = X_train[col].median()
    X_train[col] = X_train[col].fillna(mediana_train)
    X_test[col]  = X_test[col].fillna(mediana_train)

# ── 4. Definición dos modelos base ────────────────────────────────────────────
modelos_base = []

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

modelos_base.append(('grad_boost', GradientBoostingClassifier(
    n_estimators=150, learning_rate=0.05, max_depth=4, random_state=SEED
)))

modelos_base.append(('mlp', MLPClassifier(
    hidden_layer_sizes=(128, 64), max_iter=400, early_stopping=True, random_state=SEED
)))

modelos_base.append(('knn', KNeighborsClassifier(
    n_neighbors=7, weights='distance', n_jobs=-1
)))

# ── Meta-modelo ───────────────────────────────────────────────────────────────
meta_modelo = LogisticRegression(
    max_iter=1000,
    C=0.1, # Posibilidade de cambiar esto
    random_state=SEED
)

# ── 5. Adestramento por stacking ──────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"  FEATURES={NUM_FEATURES}, OUTLIERS={OUTLIERS}, LOG={TRANSFORMAR_LOG}, NORM={NORMALIZAR}")
print("=" * 70 + "\n")

# A chamada agora é moito máis limpa grazas ao teu módulo V3
stacking_clf, preds = adestrar_por_stacking(
    modelos_base=modelos_base,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    meta_modelo=meta_modelo,
    cv=CV,
    seed=SEED,
)

# ── 6. OTIMIZACIÓN DE LIMIARES (A Táctica dos Campións) ──────────────────────
print("\nBuscando o limiar óptimo para a Clase 3...")

# Obtemos as probabilidades (non a decisión final) do noso stacking
probabilidades_train = stacking_clf.predict_proba(X_train)
probabilidades_test  = stacking_clf.predict_proba(X_test)

best_f1 = 0
best_thresh = 0.25 # O estándar sería 0.25 para 4 clases

# Probamos todos os limiares entre 10% e 40%
for thresh in np.linspace(0.10, 0.40, 60):
    # Por defecto, collemos a clase con maior probabilidade
    preds_temporais = np.argmax(probabilidades_train, axis=1)
    
    # A MAXIA: Se a probabilidade da Clase 3 supera o limiar 'thresh', é Clase 3
    preds_temporais[probabilidades_train[:, 3] > thresh] = 3
    
    # Avaliamos como de bo é este limiar no noso Train
    score = f1_score(y_train, preds_temporais, average='macro')
    if score > best_f1:
        best_f1 = score
        best_thresh = thresh

print(f"Limiar óptimo atopado para Clase 3: {best_thresh:.4f}")
print(f"F1-Macro no Train despois do Thresholding: {best_f1:.4f}")

# ── 7. Aplicar a maxia ao Test e gardar ──────────────────────────────────────
# Aplicamos a mesma regra descuberta aos datos de Kaggle
preds_test_opt = np.argmax(probabilidades_test, axis=1)
preds_test_opt[probabilidades_test[:, 3] > best_thresh] = 3

nome_arquivo = f'./resultados/24-04-2026-2_f{NUM_FEATURES}.csv'

submission = pd.DataFrame({
    'ID_Cliente':   test['ID_Cliente'],
    'Target_Risco': preds_test_opt.astype(int),
})
submission.to_csv(nome_arquivo, index=False)
print(f"Arquivo de subida xerado con éxito: {nome_arquivo} ({len(submission)} filas)")