import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')

from funciones.preprocesado2x2 import preprocesar_datos

# ── 0. Configuración e Constantes ────────────────────────────────────────────
SEED = 31416
CV_FOLDS = 5

# As variables que o script de exploración demostrou que son as mellores
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

# ── 1. Carga de datos e Ordenación Temporal ──────────────────────────────────
print("=" * 60)
print("  CARGA E ORDENACIÓN TEMPORAL (OOT)")
print("=" * 60)

train_raw = pd.read_csv('./data/train.csv')
test_raw  = pd.read_csv('./data/test.csv')

# Ordenar por se acaso
train_raw['Data_Solicitude'] = pd.to_datetime(train_raw['Data_Solicitude'], errors='coerce')
train_raw = train_raw.sort_values('Data_Solicitude').reset_index(drop=True)

# ── 2. Preprocesado ──────────────────────────────────────────────────────────
X_train_full, X_test_full, y_train, CAT_FEATURES_FULL = preprocesar_datos(
    train_raw, 
    test_raw, 
    usar_ohe=False
)

print(f"\n  Filtrando as {len(TOP_FEATURES)} variables óptimas...")

X_train = X_train_full[TOP_FEATURES].copy()
X_test  = X_test_full[TOP_FEATURES].copy()

CAT_FEATURES = [c for c in CAT_FEATURES_FULL if c in TOP_FEATURES]

print(f"  Categóricas activas neste Top: {CAT_FEATURES}")

# ── 3. Validación Cruzada Temporal (Bucle Manual) ────────────────────────────
print("\n" + "=" * 60)
print("  VALIDACIÓN CRUZADA TEMPORAL (TimeSeriesSplit)")
print("=" * 60)

tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
f1_folds = []

for fold, (idx_tr, idx_val) in enumerate(tscv.split(X_train), 1):
    X_tr, y_tr   = X_train.iloc[idx_tr], y_train.iloc[idx_tr]
    X_val, y_val = X_train.iloc[idx_val], y_train.iloc[idx_val]

    modelo_cv = CatBoostClassifier(
        iterations=800, 
        learning_rate=0.04, 
        depth=6, 
        l2_leaf_reg=3,
        # auto_class_weights='Balanced', 
        random_seed=SEED, 
        verbose=0,
        early_stopping_rounds=50
    )

    # Adestramos pasando a nova lista CAT_FEATURES filtrada
    modelo_cv.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=CAT_FEATURES)

    preds = modelo_cv.predict(X_val).flatten().astype(int)
    f1 = f1_score(y_val, preds, average='macro')
    f1_folds.append(f1)
    
    print(f"  Fold {fold} F1-Macro: {f1:.5f}")

f1_medio = np.mean(f1_folds)
print(f"\n  TimeSeries CV F1-Macro Medio: {f1_medio:.4f} ± {np.std(f1_folds):.4f}")

# ── 4. Adestramento Final e Exportación ──────────────────────────────────────
print("\nAdestrando o modelo final sobre toda a liña temporal...")

modelo_final = CatBoostClassifier(
    iterations=800, 
    learning_rate=0.04, 
    depth=6, 
    l2_leaf_reg=3,
    # auto_class_weights='Balanced', 
    random_seed=SEED, 
    verbose=0
)

# Adestramento co 100% dos datos do pasado
modelo_final.fit(X_train, y_train, cat_features=CAT_FEATURES)

# Predición para o futuro (o test do kaggle)
preds_test = modelo_final.predict(X_test).flatten().astype(int)

nome_arquivo = './resultados/27-04-2026-catboost-temporal-f15.csv'
submission = pd.DataFrame({
    'ID_Cliente': test_raw['ID_Cliente'],
    'Target_Risco': preds_test,
})
submission.to_csv(nome_arquivo, index=False)
print(f"Arquivo xerado con éxito: {nome_arquivo}")