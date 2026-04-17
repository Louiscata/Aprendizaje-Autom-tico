import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# ── 1. Cargar datos ──────────────────────────────────────────────────────────
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# ── 2. Features escollidas (top correlación con target) ─────────────────────
FEATURES = [
    'Historial_Impagos',        # corr ~0.67
    'Consultas_Risco_6M',       # corr ~0.53
    'Ratio_Cota_Ingresos',      # corr ~0.53
    'Fondo_Emerxencia_Meses',   # corr ~0.43
    'Indice_Estres_Financeiro',  # corr ~0.40
]

X_train = train[FEATURES].copy()
y_train = train['Target_Risco'].copy()
X_test  = test[FEATURES].copy()

# ── 3. Imputar nulos (hai poucos, usamos mediana) ───────────────────────────
for col in FEATURES:
    median = X_train[col].median()
    X_train[col] = X_train[col].fillna(median)
    X_test[col]  = X_test[col].fillna(median)

# ── 4. Normalización ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 5. MLP ───────────────────────────────────────────────────────────────────
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    max_iter=300,
    random_state=SEED,
    early_stopping=True,
    validation_fraction=0.1,
)

# Validación cruzada rápida para ver a accuracy antes de enviar
cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Adestramento final co dataset completo
model.fit(X_train_sc, y_train)
print(f"Train Accuracy: {accuracy_score(y_train, model.predict(X_train_sc)):.4f}")

# ── 6. Predicións e arquivo de envío ─────────────────────────────────────────
preds = model.predict(X_test_sc)

submission = pd.DataFrame({
    'ID_Cliente': test['ID_Cliente'],
    'Target_Risco': preds
})
submission.to_csv('./resultados/submission_mlp.csv', index=False)
print(f"\nArquivo xerado: submission_mlp.csv  ({len(submission)} filas)")