import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import warnings

# ── Importamos as funcións do noso módulo ────────────────────────────────────
from funciones.preprocesado import preprocesar_datos, seleccion_de_variables

warnings.filterwarnings('ignore')

# ── Constantes ───────────────────────────────────────────────────────────────
SEED = 42
NUM_FEATURES = 15

np.random.seed(SEED)

# ── 1. Cargar datos ──────────────────────────────────────────────────────────
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# ── 2. Preprocesamento (Limpeza, One-Hot, Aliñamento e Imputación) ───────────
X_train_full, X_test_full, y_train = preprocesar_datos(train, test)

# ── 3. Selección de variables ────────────────────────────────────────────────
FEATURES = seleccion_de_variables(X_train_full, y_train, n=NUM_FEATURES)

X_train = X_train_full[FEATURES].copy()
X_test  = X_test_full[FEATURES].copy()

# ── 4. Modelo (Comité de Votação) ────────────────────────────────────────────
# 4.1 Definimos os modelos individuais do nosso comité
rf_model = RandomForestClassifier(
    n_estimators=100,             
    max_depth=10,                 
    class_weight='balanced',      
    n_jobs=-1,                    
    random_state=SEED
)

et_model = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    n_jobs=-1,
    random_state=SEED
)

# 4.2 Criamos o Comité agrupando ambos os modelos
# voting='soft' promedia as probabilidades em vez de contar apenas os votos diretos
model = VotingClassifier(
    estimators=[
        ('RandomForest', rf_model),
        ('ExtraTrees', et_model)
    ],
    voting='soft'
)

# Validación cruzada
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
print(f"CV F1-Macro (Comité - Top {NUM_FEATURES}): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Adestramento final
model.fit(X_train, y_train)

# Métrica no adestramento
preds_train = model.predict(X_train)
f1_train = f1_score(y_train, preds_train, average='macro')
print(f"Train F1-Macro: {f1_train:.4f}")

# ── 5. Predicións e arquivo de envío ─────────────────────────────────────────
preds = model.predict(X_test)

nome_arquivo = f'./resultados/18-04-26-voting15.csv'

submission = pd.DataFrame({
    'ID_Cliente': test['ID_Cliente'],
    'Target_Risco': preds
})
submission.to_csv(nome_arquivo, index=False)
print(f"\nArquivo xerado con éxito: {nome_arquivo} ({len(submission)} filas)")