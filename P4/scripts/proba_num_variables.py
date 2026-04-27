import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
import warnings

# Importamos o noso motor limpo
from funciones.preprocesado2x2 import preprocesar_datos

warnings.filterwarnings('ignore')

# ── 0. Configuración da Exploración ──────────────────────────────────────────
SEED = 31416
CV_FOLDS = 5
N_MIN = 10  # Número mínimo de variables a probar
N_MAX = 15

print("=" * 60)
print("  LABORATORIO: BUSCA DO NÚMERO ÓPTIMO DE VARIABLES")
print("=" * 60)

# ── 1. Carga e Preprocesado ──────────────────────────────────────────────────
train_raw = pd.read_csv('./data/train.csv')
test_raw  = pd.read_csv('./data/test.csv')

# Ordenación temporal vital
train_raw['Data_Solicitude'] = pd.to_datetime(train_raw['Data_Solicitude'], errors='coerce')
train_raw = train_raw.sort_values('Data_Solicitude').reset_index(drop=True)

X_train, X_test, y_train, CAT_FEATURES = preprocesar_datos(train_raw, test_raw, usar_ohe=False)

# ── 2. Ranking Global de Variables ───────────────────────────────────────────
print("\n" + "=" * 60)
print("  FASE 1: RANKING DE IMPORTANCIA (CatBoost)")
print("=" * 60)

# Adestramos un modelo rápido sobre todo o histórico para ver que lle gusta
modelo_ranking = CatBoostClassifier(
    iterations=300, learning_rate=0.08, depth=6, 
    auto_class_weights='Balanced', random_seed=SEED, verbose=0
)
modelo_ranking.fit(X_train, y_train, cat_features=CAT_FEATURES)

# Extraemos e ordenamos as variables
importancias = pd.Series(
    modelo_ranking.get_feature_importance(), 
    index=X_train.columns
).sort_values(ascending=False)

RANKING = importancias.index.tolist()

for i, var in enumerate(RANKING, 1):
    print(f"  {i:>2}. {var:<30} ({importancias[var]:.4f})")


# ── 3. Bucle de Exploración (A proba de lume temporal) ───────────────────────
print("\n" + "=" * 60)
print(f"  FASE 2: SIMULACIÓN OOT (De {N_MIN} a {N_MAX} variables)")
print("=" * 60)

tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
resultados = {}

for n in range(N_MIN, N_MAX + 1):
    # Collemos só as N mellores variables
    top_n_features = RANKING[:n]
    
    # Filtramos a lista de categóricas para non pasarlle a CatBoost algunha que borramos
    cat_features_n = [c for c in CAT_FEATURES if c in top_n_features]
    
    f1_folds = []
    
    # Bucle de validación temporal para este 'n'
    for idx_tr, idx_val in tscv.split(X_train):
        X_tr, y_tr   = X_train[top_n_features].iloc[idx_tr], y_train.iloc[idx_tr]
        X_val, y_val = X_train[top_n_features].iloc[idx_val], y_train.iloc[idx_val]

        # Modelo algo máis rápido para non eternizar a busca
        modelo_cv = CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6, 
            auto_class_weights='Balanced', random_seed=SEED, 
            early_stopping_rounds=40, verbose=0
        )
        
        modelo_cv.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=cat_features_n)
        
        preds = modelo_cv.predict(X_val).flatten().astype(int)
        f1_folds.append(f1_score(y_val, preds, average='macro'))

    f1_medio = np.mean(f1_folds)
    resultados[n] = f1_medio
    print(f"  ➜ N={n:>2} variables | F1-Macro: {f1_medio:.5f}")


# ── 4. Conclusión e Veredito ─────────────────────────────────────────────────
mellor_n = max(resultados, key=resultados.get)
mellor_f1 = resultados[mellor_n]
variables_a_borrar = RANKING[mellor_n:] # As que quedaron fóra do corte

print("\n" + "=" * 60)
print("  VEREDITO FINAL DA EXPLORACIÓN")
print("=" * 60)
print(f"  ✔ O número óptimo de variables é: {mellor_n}")
print(f"  ✔ F1-Macro acadado: {mellor_f1:.5f}")

if len(variables_a_borrar) > 0:
    print("\n  [ACCIÓN REQUIRIDA]")
    print(f"  MALARDAS_NOVAS = {variables_a_borrar}")
else:
    print("\n  Todas as variables están achegando valor. Non hai que borrar nada máis.")
print("=" * 60 + "\n")