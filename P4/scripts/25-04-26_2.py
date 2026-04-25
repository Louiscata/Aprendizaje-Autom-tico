import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from funciones.modeladoV3 import adestrar_por_blending, adestrar_por_stacking
from funciones.preprocesadoV3 import preprocesar_datos, seleccion_de_variables

# Importamos os motores modernos
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

import warnings
warnings.filterwarnings('ignore')

# ── 0. Variables globais ─────────────────────────────────────────────────────
SEED = 42
NUM_FEATURES = 15 # Subimos un chisco para acomodar as variables derivadas do preprocesadoV3

# DATOS CRUS: O segredo das árbores
OUTLIERS        = None   
NORMALIZAR      = False  
UMBRAL_NAN      = 10
TRANSFORMAR_LOG = False  

# ── 1. Cargar datos ───────────────────────────────────────────────────────────
train = pd.read_csv('./data/train.csv')
test  = pd.read_csv('./data/test.csv')

# ── 2. Preprocesamento (Xa inclúe a Enxeñaría de Variables internamente) ──────
# preprocesar_datos chama a crear_features(), xerando os ratios financeiros
X_train_full, X_test_full, y_train = preprocesar_datos(
    train,
    test,
    outliers=OUTLIERS,
    umbral_nan=UMBRAL_NAN,
    normalizar=NORMALIZAR,
    trans_log=TRANSFORMAR_LOG,
)

# ── 3. Selección das mellores variables (PEARSON) ─────────────────────────────
features = seleccion_de_variables(X_train_full, y_train, NUM_FEATURES)

X_train = X_train_full[features].copy()
X_test  = X_test_full[features].copy()

# Salvavidas de Nulos (Para variables derivadas ou extraídas)
for col in features:
    mediana_train = X_train[col].median()
    X_train[col] = X_train[col].fillna(mediana_train)
    X_test[col]  = X_test[col].fillna(mediana_train)

# ── 4. Definición dos modelos base ────────────────────────────────────────────
modelos_base = []

modelos_base.append(('random_forest', RandomForestClassifier(
    n_estimators=100, max_depth=10, class_weight='balanced', random_state=SEED, n_jobs=-1
)))

if LGBM_AVAILABLE:
    modelos_base.append(('lgbm', LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6, class_weight='balanced', 
        random_state=SEED, n_jobs=-1, verbose=-1
    )))

if XGB_AVAILABLE:
    modelos_base.append(('xgb', XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        eval_metric='mlogloss', random_state=SEED, n_jobs=-1, verbosity=0
    )))

meta_modelo = LogisticRegression(max_iter=1000, random_state=SEED)

# ── 5. Adestramento por Blending ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("  MODELO DE RÉCORD + VARIABLES FINANCEIRAS + BLENDING")
print("=" * 70 + "\n")


ensamblaxe, preds = adestrar_por_blending(
    modelos_base=modelos_base,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    meta_modelo=meta_modelo,
    test_size=0.20, # O 20% gárdase puro para adestrar a Loxística
    seed=SEED
)



# ── 6. Arquivo de envío ───────────────────────────────────────────────────────
nome_arquivo = f'./resultados/25-04-2026_2_blending_f{NUM_FEATURES}.csv'

submission = pd.DataFrame({
    'ID_Cliente':   test['ID_Cliente'],
    'Target_Risco': preds.astype(int),
})
submission.to_csv(nome_arquivo, index=False)
print(f"\nArquivo xerado con éxito: {nome_arquivo} ({len(submission)} filas)")