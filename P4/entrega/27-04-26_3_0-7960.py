import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
import warnings

from preprocesado import preprocesar_datos
from modelado import crear_catboost

warnings.filterwarnings('ignore')

SEED = 777
CV_FOLDS = 5
BALANCEO_CLASES = None  # Probamos con None, mellores resultados que con 'Balanced'

# Features seleccionadas manualmente tras analizar importancias dun modelo auxiliar
TOP_FEATURES = [
    'Historial_Impagos',
    'Utilizacion_Credito',
    'Consultas_Risco_6M',
    'Codigo_Postal',
    'Fondo_Emerxencia_Meses',
    'ratio_debeda_ingresos',
    'Ratio_Cota_Ingresos',
    'Indice_Estres_Financeiro',
    'Anos_Emprego',
    'Ingresos_Anuais',
    'Variacion_Saldo_6M',
    'ratio_saldo_mensual',
    'Antiguedade_Cliente_Anos',
    'Saldo_Medio_3M',
    'Distancia_Oficina_Km'
]

print("=" * 60)
print("  1. CARGA E ORDENACIÓN TEMPORAL (OOT)")
print("=" * 60)

train_raw = pd.read_csv('./data/train.csv')
test_raw  = pd.read_csv('./data/test.csv')

# Ordenamos cronoloxicamente para que o CV temporal teña sentido
train_raw['Data_Solicitude'] = pd.to_datetime(train_raw['Data_Solicitude'], errors='coerce')
train_raw = train_raw.sort_values('Data_Solicitude').reset_index(drop=True)

X_train_full, X_test_full, y_train, CAT_FEATURES_FULL = preprocesar_datos(
    train_raw, test_raw, usar_ohe=False
)

# Quedámonos só coas top features e filtramos as categóricas que estean nela
X_train = X_train_full[TOP_FEATURES].copy()
X_test  = X_test_full[TOP_FEATURES].copy()
CAT_FEATURES = [c for c in CAT_FEATURES_FULL if c in TOP_FEATURES]

print("\n" + "=" * 60)
print("  VALIDACIÓN TEMPORAL (TimeSeriesSplit)")
print("=" * 60)

# TimeSeriesSplit garante que sempre adestras co pasado e validas co futuro
tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
f1_folds = []

for fold, (idx_tr, idx_val) in enumerate(tscv.split(X_train), 1):
    X_tr, y_tr   = X_train.iloc[idx_tr], y_train.iloc[idx_tr]
    X_val, y_val = X_train.iloc[idx_val], y_train.iloc[idx_val]

    # Tres CatBoost con hiperparámetros distintos para diversificar o comité
    m1 = crear_catboost(SEED,   6, 800,  0.04, 3, balance=BALANCEO_CLASES)
    m2 = crear_catboost(SEED+1, 8, 600,  0.03, 5, balance=BALANCEO_CLASES)
    m3 = crear_catboost(SEED+2, 4, 1000, 0.05, 1, balance=BALANCEO_CLASES)

    m1.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=CAT_FEATURES)
    m2.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=CAT_FEATURES)
    m3.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=CAT_FEATURES)

    # Soft voting: promediamos as probabilidades, non os votos directos
    probs = (m1.predict_proba(X_val) + m2.predict_proba(X_val) + m3.predict_proba(X_val)) / 3
    preds = probs.argmax(axis=1)

    f1 = f1_score(y_val, preds, average='macro')
    f1_folds.append(f1)
    print(f"  Fold {fold} F1-Macro: {f1:.5f}")

print(f"\n  - F1-Macro Medio: {np.mean(f1_folds):.4f} ± {np.std(f1_folds):.4f}")

print("\nXerando predición final para Kaggle...")

# Adestramento final con todo o train, sen eval_set nin early stopping
f1 = crear_catboost(SEED,   6, 800,  0.04, 3, balance=BALANCEO_CLASES)
f2 = crear_catboost(SEED+1, 8, 600,  0.03, 5, balance=BALANCEO_CLASES)
f3 = crear_catboost(SEED+2, 4, 1000, 0.05, 1, balance=BALANCEO_CLASES)

f1.fit(X_train, y_train, cat_features=CAT_FEATURES)
f2.fit(X_train, y_train, cat_features=CAT_FEATURES)
f3.fit(X_train, y_train, cat_features=CAT_FEATURES)

probs_final = (f1.predict_proba(X_test) + f2.predict_proba(X_test) + f3.predict_proba(X_test)) / 3
preds_final = probs_final.argmax(axis=1)

submission = pd.DataFrame({'ID_Cliente': test_raw['ID_Cliente'], 'Target_Risco': preds_final})
nome_arquivo = './resultados/27-04-26_3.csv'
submission.to_csv(nome_arquivo, index=False)
print(f"Arquivo xerado: {nome_arquivo}")