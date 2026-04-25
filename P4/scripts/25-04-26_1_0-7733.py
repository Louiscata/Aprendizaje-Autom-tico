import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from funciones.preprocesadoV3 import preprocesar_datos, seleccion_de_variables

import warnings
warnings.filterwarnings('ignore')

# ── 0. Variables globais (VOLVEMOS Á PUREZA DO 0.7912) ──────────────────────
SEED            = 42
NUM_FEATURES    = 20     

# Datos completamente crus
OUTLIERS        = None   
NORMALIZAR      = False  
UMBRAL_NAN      = 10
TRANSFORMAR_LOG = False  

# ── 1. Cargar datos ───────────────────────────────────────────────────────────
train = pd.read_csv('./data/train.csv')
test  = pd.read_csv('./data/test.csv')

# ── 2. Preprocesamento e Selección (O Récord Orixinal) ────────────────────────
X_train_full, X_test_full, y_train = preprocesar_datos(
    train, 
    test, 
    outliers=OUTLIERS, 
    umbral_nan=UMBRAL_NAN,
    normalizar=NORMALIZAR, 
    trans_log=TRANSFORMAR_LOG,
)

features = seleccion_de_variables(X_train_full, y_train, NUM_FEATURES)

X_train = X_train_full[features].copy()
X_test  = X_test_full[features].copy()

# Salvavidas de Nulos
for col in features:
    mediana_train = X_train[col].median()
    X_train[col] = X_train[col].fillna(mediana_train)
    X_test[col]  = X_test[col].fillna(mediana_train)

# ── 3. Os 3 Motores Orixinais do Récord ───────────────────────────────────────
modelos_base = []

# 1. Random Forest (Orixinal)
modelos_base.append(('rf', RandomForestClassifier(
    n_estimators=100, max_depth=10, class_weight='balanced', random_state=SEED, n_jobs=-1
)))

# 2. LightGBM (Orixinal)
if LGBM_AVAILABLE:
    modelos_base.append(('lgbm', LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6, class_weight='balanced',
        random_state=SEED, n_jobs=-1, verbose=-1
    )))

# 3. XGBoost (Orixinal)
if XGB_AVAILABLE:
    modelos_base.append(('xgb', XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        eval_metric='mlogloss', random_state=SEED, n_jobs=-1, verbosity=0
    )))

# ── 4. A Nova Maxia: Voting Classifier (Sen Meta-Aprendizaxe) ────────────────
print("\n" + "=" * 70)
print("  ENSAMBLAXE VOTING (SOFT) - PREVIR O SOBREAXUSTE")
print("=" * 70 + "\n")

# voting='soft' fai a media das probabilidades exactas dos 3 modelos.
ensamblaxe_voting = VotingClassifier(
    estimators=modelos_base,
    voting='soft',
    n_jobs=-1
)

# ── 5. Validación Cruzada e Adestramento ─────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print("Executando validación cruzada do Voting...")
cv_scores = cross_val_score(ensamblaxe_voting, X_train, y_train, cv=skf, scoring='f1_macro')
print(f"  CV F1-Macro (Voting): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

print("Adestramento final...")
ensamblaxe_voting.fit(X_train, y_train)

preds_train = ensamblaxe_voting.predict(X_train)
f1_train = f1_score(y_train, preds_train, average='macro')
print(f"  Train F1-Macro: {f1_train:.4f}")

# ── 6. Arquivo de envío ───────────────────────────────────────────────────────
preds_test = ensamblaxe_voting.predict(X_test)

nome_arquivo = f'./resultados/25-04-2026-VOTING_SOFT_f{NUM_FEATURES}.csv'
submission = pd.DataFrame({
    'ID_Cliente':   test['ID_Cliente'],
    'Target_Risco': preds_test.astype(int),
})
submission.to_csv(nome_arquivo, index=False)
print(f"\nArquivo de subida xerado con éxito: {nome_arquivo}")