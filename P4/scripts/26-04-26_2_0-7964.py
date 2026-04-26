import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from funciones.modelado_old import adestrar_por_stacking
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

# ── 0. Variables globais ─────────────────────────────────────────────────────
SEED = 777
NUM_FEATURES = 18

# ── 1. Cargar datos ──────────────────────────────────────────────────────────
train = pd.read_csv('./data/train.csv')
test  = pd.read_csv('./data/test.csv')
 
# ── 2. Preprocesamento inicial (Codificación e limpeza) ──────────────────────
y_train = train['Target_Risco'].copy()
 
X_train_raw = train.drop(columns=['ID_Cliente', 'Target_Risco', 'Data_Solicitude'])
X_test_raw  = test.drop(columns=['ID_Cliente', 'Data_Solicitude'])
 
X_train_encoded = pd.get_dummies(X_train_raw)
X_test_encoded  = pd.get_dummies(X_test_raw)
 
X_train_encoded, X_test_encoded = X_train_encoded.align(
    X_test_encoded, join='left', axis=1, fill_value=0
)
 
# ── 3. Selección de variables ───────────────────────────────────────────────
df_corr = X_train_encoded.copy()
df_corr['_target_'] = y_train.values
correlacions = df_corr.corr()['_target_'].abs().sort_values(ascending=False)
FEATURES = correlacions.index[1:NUM_FEATURES + 1].tolist()
 
print(f"--- TOP {NUM_FEATURES} Variables (Pearson) ---")
for feat in FEATURES:
    print(f"  · {feat}  (corr: {correlacions[feat]:.4f})")
print("-" * 35 + "\n")

X_train = X_train_encoded[FEATURES].copy()
X_test  = X_test_encoded[FEATURES].copy()
 
# ── 4. Imputar nulos (mediana) ───────────────────────────────────────────────
for col in FEATURES:
    median = X_train[col].median()
    X_train[col] = X_train[col].fillna(median)
    X_test[col]  = X_test[col].fillna(median)
 
# ── 5. Definición dos modelos base ───────────────────────────────────────────
modelos_base = []

# modelos_base.append(('random_forest', RandomForestClassifier(
#     n_estimators=100, max_depth=10, class_weight='balanced', random_state=SEED, n_jobs=-1
# )))

modelos_base.append(('lgbm', LGBMClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=6, class_weight='balanced', 
    random_state=SEED, n_jobs=-1, verbose=-1
)))

modelos_base.append(('xgb', XGBClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=5,
    eval_metric='mlogloss', random_state=SEED, n_jobs=-1, verbosity=0
)))

modelos_base.append(('catboost', CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    auto_class_weights='Balanced',
    random_seed=SEED,
    verbose=0
)))

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
nome_arquivo = f'./resultados/26-04-2026-CatBoost-Stacking-Viejo-f{NUM_FEATURES}.csv'
 
submission = pd.DataFrame({
    'ID_Cliente':   test['ID_Cliente'],
    'Target_Risco': preds,
})
submission.to_csv(nome_arquivo, index=False)
print(f"Arquivo xerado con éxito: {nome_arquivo} ({len(submission)} filas)")