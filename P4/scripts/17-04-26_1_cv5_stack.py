import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

from funciones.preprocesado import seleccion_de_variables
from funciones.modelado import adestrar_por_stacking

import warnings
warnings.filterwarnings('ignore')

# ── 0. Variables globais ─────────────────────────────────────────────────────

SEED = 42
NUM_FEATURES = 20

# ── 1. Cargar datos ──────────────────────────────────────────────────────────
train = pd.read_csv('./data/train.csv')
test  = pd.read_csv('./data/test.csv')
 
# ── 2. Preprocesamento inicial (Codificación e limpeza) ──────────────────────
y_train = train['Target_Risco'].copy()
 
X_train_raw = train.drop(columns=['ID_Cliente', 'Target_Risco', 'Data_Solicitude'])
X_test_raw  = test.drop(columns=['ID_Cliente', 'Data_Solicitude'])
 
X_train_encoded = pd.get_dummies(X_train_raw)
X_test_encoded  = pd.get_dummies(X_test_raw)
 
# Aliñamos por se hai categorías distintas entre train e test
X_train_encoded, X_test_encoded = X_train_encoded.align(
    X_test_encoded, join='left', axis=1, fill_value=0
)
 
# ── 3. Selección de variables ─────────────────────────────────────────────────
FEATURES = seleccion_de_variables(X_train_encoded, y_train, NUM_FEATURES)
 
X_train = X_train_encoded[FEATURES].copy()
X_test  = X_test_encoded[FEATURES].copy()
 
# ── 4. Imputar nulos (mediana) ───────────────────────────────────────────────
for col in FEATURES:
    median = X_train[col].median()
    X_train[col] = X_train[col].fillna(median)
    X_test[col]  = X_test[col].fillna(median)
 
# ── 5. Definición dos modelos base e do meta-modelo ──────────────────────────
#
#   Usamos modelos heteroxéneos para maximizar a diversidade das predicións:
#     · Random Forest      → bo con features non lineais, robusto a ruído
#     · Gradient Boosting  → forte en precisión, complementa ao RF
#     · MLP                → captura patróns non lineais complexos
#     · KNN                → basado en distancias, perspectiva moi diferente
#
#   Meta-modelo: Regresión Loxística (simple e interpretable; recibe as
#   probabilidades dos base e aprende a ponderalas)
 
modelos_base = [
    ('random_forest', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1,
    )),
    ('gradient_boosting', GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=SEED,
    )),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=300,
        random_state=SEED,
    )),
    ('knn', KNeighborsClassifier(
        n_neighbors=7,
        n_jobs=-1,
    )),
]
 
meta_modelo = LogisticRegression(max_iter=1000, random_state=SEED)
 
# ── 6. Adestramento por stacking ─────────────────────────────────────────────
stacking_clf, preds = adestrar_por_stacking(
    modelos_base=modelos_base,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    meta_modelo=meta_modelo,
    cv=5,
)
 
# ── 7. Arquivo de envío ──────────────────────────────────────────────────────
nome_arquivo = f'./resultados/17-04-2026-stacking-features{NUM_FEATURES}.csv'
 
submission = pd.DataFrame({
    'ID_Cliente':   test['ID_Cliente'],
    'Target_Risco': preds,
})
submission.to_csv(nome_arquivo, index=False)
print(f"Arquivo xerado con éxito: {nome_arquivo} ({len(submission)} filas)")