import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# ── 1. Cargar datos ──────────────────────────────────────────────────────────
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# ── 2. Preprocesamento inicial (Codificación de categóricas) ─────────────────
y_train = train['Target_Risco'].copy()

# Eliminamos o ID e o Target para preparar as variables preditoras
X_train_raw = train.drop(columns=['ID_Cliente', 'Target_Risco'])
X_test_raw = test.drop(columns=['ID_Cliente'])

# Aplicamos One-Hot Encoding (converte texto en múltiples columnas numéricas 0/1)
X_train_encoded = pd.get_dummies(X_train_raw)
X_test_encoded = pd.get_dummies(X_test_raw)

# Aliñamos os datasets por se algunha categoría de texto só existe no train e non no test (ou viceversa)
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

# ── 3. Selección das 10 variables con máis correlación ──────────────────────
# Engadimos o target temporalmente só para calcular as correlacións
df_corr = X_train_encoded.copy()
df_corr['Target_Risco'] = y_train

# Calculamos a correlación absoluta respecto ao Target_Risco e ordenamos de maior a menor
correlacions = df_corr.corr()['Target_Risco'].abs().sort_values(ascending=False)

# A primeira correlación é sempre 1.0 (Target_Risco consigo mesmo). 
# Collemos as 10 seguintes (do índice 1 ao 11).
FEATURES = correlacions.index[1:11].tolist()

print("--- TOP 10 Variables con maior correlación ---")
for feat in FEATURES:
    print(f" - {feat} (Correlación absoluta: {correlacions[feat]:.4f})")
print("----------------------------------------------\n")

# Filtramos os nosos datos para usar unicamente estas 10 variables gañadoras
X_train = X_train_encoded[FEATURES].copy()
X_test  = X_test_encoded[FEATURES].copy()

# ── 4. Imputar nulos (mediana) ──────────────────────────────────────────────
for col in FEATURES:
    median = X_train[col].median()
    X_train[col] = X_train[col].fillna(median)
    X_test[col]  = X_test[col].fillna(median)

# ── 5. Normalización (Crucial para o MLP) ────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 6. Modelo (MLPClassifier) ────────────────────────────────────────────────
model = MLPClassifier(
    hidden_layer_sizes=(128, 64), # Aumentamos lixeiramente as neuronas xa que agora hai máis variables
    activation='relu',
    max_iter=300,
    random_state=SEED,
    early_stopping=True,
    validation_fraction=0.1,
)

# Validación cruzada para avaliar rendemento interno
cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='f1_macro')
print(f"CV F1-Macro (Top 10 features): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Adestramento co 100% dos datos do train
model.fit(X_train_sc, y_train)

# Métrica no dataset de adestramento para vixiar o sobreaxuste (overfitting)
preds_train = model.predict(X_train_sc)
f1_train = f1_score(y_train, preds_train, average='macro')
print(f"Train F1-Macro: {f1_train:.4f}")

# ── 7. Predicións e arquivo de envío ─────────────────────────────────────────
preds = model.predict(X_test_sc)

submission = pd.DataFrame({
    'ID_Cliente': test['ID_Cliente'],
    'Target_Risco': preds
})
submission.to_csv('./resultados/13-04-26_3.csv', index=False)
print(f"\nArquivo xerado con éxito: 13-04-26_3.csv ({len(submission)} filas)")