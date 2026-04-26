import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')

# ── 1. Cargar Datos ───────────────────────────────────────────────────────────
train = pd.read_csv('./data/train.csv')
test  = pd.read_csv('./data/test.csv')

y_train = train['Target_Risco'].copy()
X_train = train.drop(columns=['ID_Cliente', 'Target_Risco', 'Data_Solicitude'])
X_test  = test.drop(columns=['ID_Cliente', 'Data_Solicitude'])

# ── 2. Preparar as variables categóricas (O Segredo) ──────────────────────────
# Convertemos os Códigos Postais a texto para que non os trate como números
if 'Codigo_Postal' in X_train.columns:
    X_train['Codigo_Postal'] = 'CP_' + X_train['Codigo_Postal'].astype(str)
    X_test['Codigo_Postal']  = 'CP_' + X_test['Codigo_Postal'].astype(str)

# Identificamos todas as columnas que son de texto (categóricas)
columnas_categoricas = X_train.select_dtypes(include=['object']).columns.tolist()

# CatBoost non admite NaNs nas categóricas por defecto, enchemos cunha palabra
for col in columnas_categoricas:
    X_train[col] = X_train[col].fillna("Descoñecido")
    X_test[col]  = X_test[col].fillna("Descoñecido")

# Para as numéricas, enchemos coa mediana
columnas_numericas = X_train.select_dtypes(exclude=['object']).columns.tolist()
for col in columnas_numericas:
    mediana = X_train[col].median()
    X_train[col] = X_train[col].fillna(mediana)
    X_test[col]  = X_test[col].fillna(mediana)

# ── 3. O Motor "CatBoost Solo" a Tope ─────────────────────────────────────────
print("Iniciando adestramento de CatBoost nativo...")

# Configuración "a tope" pero con control de sobreaxuste (L2 regularization)
modelo_cb = CatBoostClassifier(
    iterations=800,              # Suficientes árbores para aprender profundo
    learning_rate=0.04,          # Ritmo lento e seguro
    depth=7,                     # Árbores lixeiramente máis profundas
    l2_leaf_reg=3,               # Regularización para evitar memorizar Kaggle
    auto_class_weights='Balanced', # Vital para cazar as clases 1 e 3
    cat_features=columnas_categoricas, # A SÚA ARMA SECRETA
    random_seed=31416,           # A túa semente da sorte
    verbose=100                  # Só imprime cada 100 árbores
)

# Adestramos directamente con todos os datos
modelo_cb.fit(X_train, y_train)

# ── 4. Predición e Envío ──────────────────────────────────────────────────────
preds = modelo_cb.predict(X_test)
# CatBoost a veces devolve un array 2D, aplanámolo por se acaso
preds = preds.flatten()

nome_arquivo = './resultados/26-04-2026-Catboost.csv'

submission = pd.DataFrame({
    'ID_Cliente': test['ID_Cliente'],
    'Target_Risco': preds.astype(int),
})
submission.to_csv(nome_arquivo, index=False)
print(f"Arquivo xerado con éxito: {nome_arquivo}")