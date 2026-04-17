import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# ── 1. Cargar datos ──────────────────────────────────────────────────────────
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# ── 2. Features escollidas (top correlación con target) ─────────────────────
FEATURES = [
    'Historial_Impagos',
    'Consultas_Risco_6M',
    'Ratio_Cota_Ingresos',
    'Fondo_Emerxencia_Meses',
    'Indice_Estres_Financeiro',
]

X_train = train[FEATURES].copy()
y_train = train['Target_Risco'].copy()
X_test  = test[FEATURES].copy()

# ── 3. Imputar nulos (mediana) ──────────────────────────────────────────────
for col in FEATURES:
    median = X_train[col].median()
    X_train[col] = X_train[col].fillna(median)
    X_test[col]  = X_test[col].fillna(median)

# ── 4. Normalización (non se fai) ─────────────────────────────────────────────

# ── 5. Modelo (Árbore de Decisión) ───────────────────────────────────────────
model = DecisionTreeClassifier(
    max_depth=7,                  # Limitamos a profundidade para non sobreaxustar (overfitting)
    min_samples_split=20,         # Mínimo de mostras para dividir un nodo
    class_weight='balanced',      # Moi importante para maximizar o F1-Macro se hai desbalanceo
    random_state=SEED
)

# Validación cruzada usando F1-Macro (a métrica oficial da competición)
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
print(f"CV F1-Macro (Decision Tree): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Adestramento final co dataset completo
model.fit(X_train, y_train)

# Medimos o F1-Macro no propio adestramento para comprobar se hai overfitting
preds_train = model.predict(X_train)
f1_train = f1_score(y_train, preds_train, average='macro')
print(f"Train F1-Macro (Decision Tree): {f1_train:.4f}")

# ── 6. Predicións e arquivo de envío ─────────────────────────────────────────
preds = model.predict(X_test)

submission = pd.DataFrame({
    'ID_Cliente': test['ID_Cliente'],
    'Target_Risco': preds
})
submission.to_csv('./resultados/13-04-26_2.csv', index=False)
print(f"\nArquivo xerado: 13-04-26_2.csv  ({len(submission)} filas)")