import pandas as pd
import numpy as np

from sklearn.ensemble import (
    ExtraTreesClassifier, 
    RandomForestClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Importamos LightGBM (o rei dos datos sen procesar)
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("[AVISO] lightgbm non instalado. Executar: pip install lightgbm")

from funciones.preprocesado import preprocesar_datos, seleccion_de_variables
from funciones.modelado import adestrar_por_stacking

import warnings
warnings.filterwarnings('ignore')

# ── 0. Variables globais ───────────────────────────────────
SEED            = 42
NUM_FEATURES    = 18
CV              = 5

# A maxia está aquí: Deixar os datos crus para as árbores
OUTLIERS        = None
NORMALIZAR      = True    # Axuda ao MLP e ao KNN, as árbores ignórano
UMBRAL_NAN      = 10
TRANSFORMAR_LOG = False   # NON suavizar os cartos, que as árbores vexan os extremos

# ── 1. Cargar datos ───────────────────────────────────────────────────────────
train = pd.read_csv('./data/train.csv')
test  = pd.read_csv('./data/test.csv')

# ── 2. Preprocesamento ────────────────────────────────────────────────────────
# Lembra adaptar o nome dos parámetros (ex: modo_outliers ou eliminar_outliers)
# segundo como o teñas escrito no teu preprocesado.py
X_train_full, X_test_full, y_train = preprocesar_datos(
    train,
    test,
    outliers=OUTLIERS, 
    umbral_nan=UMBRAL_NAN,
    normalizar=NORMALIZAR,
    trans_log=TRANSFORMAR_LOG,
)

# ── 3. Selección das mellores variables (Mutual Information) ──────────────────
features = seleccion_de_variables(X_train_full, y_train, NUM_FEATURES)

X_train = X_train_full[features].copy()
X_test  = X_test_full[features].copy()

# ── 4. Definición dos modelos base (Os 4 Fantásticos) ─────────────────────────
modelos_base = []

# 1. O motor principal: LightGBM (Se non o tes, cambia a XGBoost)
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

# 2. O apoio de varianza: Extra Trees
modelos_base.append(('extra_trees', ExtraTreesClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight='balanced',
    min_samples_leaf=2,
    random_state=SEED,
    n_jobs=-1,
)))

# 3. O modelo matemático complexo: Redes Neuronais (MLP)
modelos_base.append(('mlp', MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=SEED,
)))

# 4. O modelo espacial: K-Nearest Neighbors
modelos_base.append(('knn', KNeighborsClassifier(
    n_neighbors=5,
    n_jobs=-1,
    weights='distance',
)))

# ── Meta-modelo ───────────────────────────────────────────────────────────────
# Usamos un Random Forest moi rasiño (max_depth=3). 
# Ao ser tan simple, non memorizará, só aprenderá en que modelo confiar para cada caso.
meta_modelo = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=SEED
)

# ── 5. Adestramento por stacking ──────────────────────────────────────────────
stacking_clf, preds = adestrar_por_stacking(
    modelos_base=modelos_base,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    meta_modelo=meta_modelo,
    cv=CV,
    passthrough=False,    # Fundamental: Que o meta-modelo só vexa as probabilidades
    seed=SEED,
)

# ── 6. Arquivo de envío ───────────────────────────────────────────────────────
nome_arquivo = f'./resultados/22-04-2026_stacking_5_f{NUM_FEATURES}.csv'

submission = pd.DataFrame({
    'ID_Cliente':   test['ID_Cliente'],
    'Target_Risco': preds.astype(int),
})
submission.to_csv(nome_arquivo, index=False)
print(f"Arquivo xerado: {nome_arquivo} ({len(submission)} filas)")