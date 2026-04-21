import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# LightGBM e XGBoost son os mellores modelos para datos tabulares en Kaggle.
# Instalar con: pip install lightgbm xgboost
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
NUM_FEATURES  = 25    # aumentado: MI + feature eng. engade variables útiles
CV            = 5

ELIMINAR_OUTLIERS = None
NORMALIZAR        = True
UMBRAL_NAN        = 10

# ── 1. Cargar datos ───────────────────────────────────────────────────────────
train = pd.read_csv('./data/train.csv')
test  = pd.read_csv('./data/test.csv')

# ── 2. Preprocesamento completo ───────────────────────────────────────────────
# Inclúe: limpeza, feature engineering, OHE, Codigo_Postal como categórica,
# extracción de features temporais de Data_Solicitude, e normalización.
X_train_full, X_test_full, y_train = preprocesar_datos(
    train,
    test,
    outliers=ELIMINAR_OUTLIERS,
    umbral_nan=UMBRAL_NAN,
    normalizar=NORMALIZAR,
)

# ── 3. Selección das mellores variables (Mutual Information) ──────────────────
# [FIX] Usamos MI en lugar de correlación de Pearson. MI captura dependencias
# non-lineais e é o método correcto para clasificación multiclase.
features = seleccion_de_variables(X_train_full, y_train, NUM_FEATURES)

X_train = X_train_full[features].copy()
X_test  = X_test_full[features].copy()

# ── 4. Definición dos modelos base e do meta-modelo ──────────────────────────
#
# [FIX] Todos os modelos teñen class_weight='balanced' onde é soportado.
# [NOVO] LightGBM e XGBoost como modelos base — son os máis potentes para
#        datos tabulares e habitualmente gañan competicións Kaggle.
#
modelos_base = []

# LightGBM — moi rápido, excelente con features mixtas e clases desbalanceadas
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

# XGBoost — complementa LightGBM con outra estratexia de boosting
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

# ExtraTreesClassifier — alta varianza, bó para diversidade no ensemble
modelos_base.append(('extra_trees', ExtraTreesClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight='balanced',      # [FIX] xa estaba
    min_samples_leaf=2,
    random_state=SEED,
    n_jobs=-1,
)))

# GradientBoostingClassifier de sklearn — non ten class_weight, compensamos
# con sample_weight no adestramento ou con max_features para regularizar
modelos_base.append(('gradient_boosting', GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=5,
    random_state=SEED,
)))

# SVC — complementa os árbores ao ter un límite de decisión diferente
modelos_base.append(('svc', SVC(
    kernel='rbf',
    probability=True,
    class_weight='balanced',      # [FIX] xa estaba
    random_state=SEED,
    C=1.0,
)))

# MLP — bo para capturar interaccións complexas
# [FIX] engadimos class_weight manualmente a través de sample_weight
# non soportado directamente en MLPClassifier; como alternativa,
# aumentamos capacidade e usamos early_stopping para non sobreaxustar
modelos_base.append(('mlp', MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=SEED,
)))

# KNN — útil como modelo de "veciñanza" que outros non capturan
# [FIX] KNN non ten class_weight; con datos desbalanceados pode ignorar
# clases minoritarias. Reducimos k para ser máis sensible a elas.
modelos_base.append(('knn', KNeighborsClassifier(
    n_neighbors=5,    # [FIX] reducido de 7 a 5 para ser máis sensible
    n_jobs=-1,
    weights='distance',   # ponderar por distancia mellora con desbalanceo
)))

# ── Meta-modelo ───────────────────────────────────────────────────────────────
# [FIX] class_weight='balanced' para non ignorar as clases minoritarias.
# [FIX] passthrough=True no stacking: o meta-modelo verá tamén as features
#       orixinais, non só as saídas dos modelos base.
meta_modelo = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    C=0.5,             # algo de regularización para evitar sobreaxuste
    random_state=SEED,
)

# ── 5. Adestramento por stacking ─────────────────────────────────────────────
stacking_clf, preds = adestrar_por_stacking(
    modelos_base=modelos_base,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    meta_modelo=meta_modelo,
    cv=CV,
    passthrough=True,    # [FIX] o meta-modelo ve as features orixinais
    seed=SEED,
)

# ── 6. Arquivo de envío ───────────────────────────────────────────────────────
nome_arquivo = f'./resultados/21-04-2026_2_stacking_f{NUM_FEATURES}.csv'

submission = pd.DataFrame({
    'ID_Cliente':   test['ID_Cliente'],
    'Target_Risco': np.round(preds).astype(int),
})
submission.to_csv(nome_arquivo, index=False)
print(f"Arquivo xerado: {nome_arquivo} ({len(submission)} filas)")