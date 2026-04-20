import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score

from funciones.preprocesado import seleccion_de_variables, preprocesar_datos
from funciones.modelado import adestrar_por_boosting

import warnings
warnings.filterwarnings('ignore')

# ── 0. Variables globais ─────────────────────────────────────────────────────

SEED = 42
N    = 15
CV   =  5
TIPO = 'gradient'    # 'gradient' ou 'adaboost'

# ── 1. Cargar datos ──────────────────────────────────────────────────────────
train = pd.read_csv('./data/train.csv')
test  = pd.read_csv('./data/test.csv')

# ── 2. Preprocesamento ───────────────────────────────────────────────────────
X_train_encoded, X_test_encoded, y_train = preprocesar_datos(
    train=train,
    test=test,
    eliminar_outliers=True,
    umbral_nan=10,
    normalizar=True,
)

# ── 3. Hiperparámetros do boosting ───────────────────────────────────────────
BOOSTING_PARAMS = dict(
    tipo          = TIPO,
    n_estimators  = 150,
    learning_rate = 0.05,
    max_depth     = 4,
)

# ── 4. Selección de variables ─────────────────────────────────────────────────
features = seleccion_de_variables(X_train_encoded, y_train, N)

X_train_sel = X_train_encoded[features].copy()
X_test_sel  = X_test_encoded[features].copy()

for col in features:
    median = X_train_sel[col].median()
    X_train_sel[col] = X_train_sel[col].fillna(median)
    X_test_sel[col]  = X_test_sel[col].fillna(median)

# ── 5. Adestramento ───────────────────────────────────────────────────────────
print(f"Adestramento con n={N} variables ({TIPO})...")
boosting_clf, preds = adestrar_por_boosting(
    X_train=X_train_sel,
    y_train=y_train,
    X_test=X_test_sel,
    **BOOSTING_PARAMS,
    cv=CV,
    seed=SEED,
)

# ── 6. Arquivo de envío ──────────────────────────────────────────────────────
nome_arquivo = f'./resultados/20-04-2026_1_boosting-{TIPO}_features{N}.csv'

submission = pd.DataFrame({
    'ID_Cliente':   test['ID_Cliente'],
    'Target_Risco': preds,
})
submission.to_csv(nome_arquivo, index=False)
print(f"\nArquivo xerado con éxito: {nome_arquivo} ({len(submission)} filas)")