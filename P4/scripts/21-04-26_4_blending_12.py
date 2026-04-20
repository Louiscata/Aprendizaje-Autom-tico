import pandas as pd
import numpy as np

from sklearn.ensemble import (
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ── Importamos as funcións do noso módulo ────────────────────────────────────
from funciones.preprocesado import preprocesar_datos, seleccion_de_variables
from funciones.modelado import adestrar_por_blending

import warnings
warnings.filterwarnings('ignore')

# ── 0. Variables globais ──────────────────────────────────────────────────────
SEED          = 42
NUM_FEATURES  = 12
TEST_SIZE     = 0.20

ELIMINAR_OUTLIERS = True
NORMALIZAR        = True
UMBRAL_NAN        = 10

# ── 1. Cargar datos ───────────────────────────────────────────────────────────
train = pd.read_csv('./data/train.csv')
test  = pd.read_csv('./data/test.csv')

# ── 2. Preprocesamento completo ───────────────────────────────────────────────
X_train_full, X_test_full, y_train = preprocesar_datos(
    train,
    test,
    eliminar_outliers=ELIMINAR_OUTLIERS,
    umbral_nan=UMBRAL_NAN,
    normalizar=NORMALIZAR,
)

# ── 3. Selección das mellores variables ───────────────────────────────────────
features = seleccion_de_variables(X_train_full, y_train, NUM_FEATURES)

X_train = X_train_full[features].copy()
X_test  = X_test_full[features].copy()

# ── 4. Definición dos modelos base e do meta-modelo ──────────────────────────
modelos_base = [
    ('extra_trees', ExtraTreesClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1,
    )),
    ('gradient_boosting', GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=SEED,
    )),
    ('svc', SVC(
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=SEED,
    )),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=400,
        early_stopping=True,
        random_state=SEED,
    )),
    ('knn', KNeighborsClassifier(
        n_neighbors=7,
        n_jobs=-1,
    )),
]

meta_modelo = LogisticRegression(max_iter=1000, random_state=SEED, class_weight='balanced')

# ── 5. Adestramento por Blending ─────────────────────────────────────────────
blending_clf, preds = adestrar_por_blending(
    modelos_base=modelos_base,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    meta_modelo=meta_modelo,
    test_size=TEST_SIZE,
    seed=SEED
)

# ── 6. Arquivo de envío ───────────────────────────────────────────────────────
nome_arquivo = f'./resultados/20-04-2026_4_blending_n{NUM_FEATURES}.csv'

submission = pd.DataFrame({
    'ID_Cliente':   test['ID_Cliente'],
    'Target_Risco': np.round(preds).astype(int),
})
submission.to_csv(nome_arquivo, index=False)
print(f"\nArquivo xerado con éxito: {nome_arquivo} ({len(submission)} filas)")