import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

# ── Constantes ───────────────────────────────────────────────────────────────
SEED = 42
NUM_FEATURES = 25

# Resultados correlación por num features
#   7   -    0.669
#   10  -    0.717
#   15  -    0.767
#   25  -    0.764

# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(SEED)

# ── 1. Cargar datos ──────────────────────────────────────────────────────────
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# ── 2. Preprocesamento inicial (Codificación e limpeza) ──────────────────────
y_train = train['Target_Risco'].copy()

# Eliminamos o ID, o Target e a Data de Solicitude
X_train_raw = train.drop(columns=['ID_Cliente', 'Target_Risco', 'Data_Solicitude'])
X_test_raw = test.drop(columns=['ID_Cliente', 'Data_Solicitude'])

# Aplicamos One-Hot Encoding (converte texto en múltiples columnas numéricas 0/1)
X_train_encoded = pd.get_dummies(X_train_raw)
X_test_encoded = pd.get_dummies(X_test_raw)

# Aliñamos os datasets por se hai categorías distintas entre train e test
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

# ── 3. Selección das variables con máis correlación ──────────────────────────
df_corr = X_train_encoded.copy()
df_corr['Target_Risco'] = y_train

# Calculamos correlacións e ordenamos de maior a menor
correlacions = df_corr.corr()['Target_Risco'].abs().sort_values(ascending=False)

# Collemos dende o 1 ata NUM_FEATURES + 1 (o índice 0 é a correlación do Target consigo mesmo, que é 1.0)
FEATURES = correlacions.index[1:NUM_FEATURES+1].tolist()

print(f"--- TOP {NUM_FEATURES} Variables ---")
for feat in FEATURES:
    print(f" - {feat} (Correlación: {correlacions[feat]:.4f})")
print("-" * 30 + "\n")

# Filtramos os datasets coas variables seleccionadas
X_train = X_train_encoded[FEATURES].copy()
X_test  = X_test_encoded[FEATURES].copy()

# ── 4. Imputar nulos (mediana) ──────────────────────────────────────────────
for col in FEATURES:
    median = X_train[col].median()
    X_train[col] = X_train[col].fillna(median)
    X_test[col]  = X_test[col].fillna(median)

# ── 5. Modelo (Random Forest) ────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=100,             
    max_depth=10,                 
    class_weight='balanced',      
    n_jobs=-1,                    
    random_state=SEED
)

# Validación cruzada
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
print(f"CV F1-Macro (Random Forest - Top {NUM_FEATURES}): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Adestramento final
model.fit(X_train, y_train)

# Métrica no adestramento
preds_train = model.predict(X_train)
f1_train = f1_score(y_train, preds_train, average='macro')
print(f"Train F1-Macro: {f1_train:.4f}")

# ── 6. Predicións e arquivo de envío ─────────────────────────────────────────
preds = model.predict(X_test)

# Nome dinámico para o arquivo CSV en base ao número de variables
nome_arquivo = f'./resultados/15-04-26-features{NUM_FEATURES}.csv'

submission = pd.DataFrame({
    'ID_Cliente': test['ID_Cliente'],
    'Target_Risco': preds
})
submission.to_csv(nome_arquivo, index=False)
print(f"\nArquivo xerado con éxito: {nome_arquivo} ({len(submission)} filas)")