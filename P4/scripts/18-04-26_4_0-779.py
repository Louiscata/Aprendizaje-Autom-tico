import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

# ── 1. Cargar datos ──────────────────────────────────────────────────────────
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# ── 2. Separar características (Eliminamos a data por exceso de categorías) ──
X_train = train.drop(columns=['ID_Cliente', 'Target_Risco', 'Data_Solicitude'])
y_train = train['Target_Risco'].copy()
X_test  = test.drop(columns=['ID_Cliente', 'Data_Solicitude'])

# ── 3. O "Truco" do Non-Preprocesado (Converter texto a Category) ────────────
# HistGradientBoosting precisa saber cales son categóricas.
# Só temos que pasalas ao tipo 'category' de Pandas. Non facemos Dummies/One-Hot!
columnas_texto = X_train.select_dtypes(include=['object']).columns

for col in columnas_texto:
    X_train[col] = X_train[col].astype('category')
    X_test[col]  = X_test[col].astype('category')

# ── 4. Balanceo de clases para maximizar o F1-Macro ──────────────────────────
# Calculamos canto pesa cada mostra para darlle máis importancia ás clases raras
pesos_mostras = compute_sample_weight(class_weight='balanced', y=y_train)

# ── 5. O Modelo Besta (HistGradientBoosting) ─────────────────────────────────
model = HistGradientBoostingClassifier(
    categorical_features='from_dtype', # Dille ao modelo que use os tipos 'category'
    max_iter=500,                      # Como non hai límite de tempo, subimos as iteracións
    learning_rate=0.05,                # Aprendizaxe lenta para xeneralizar mellor
    max_depth=10,                      # Profundidade moderada
    random_state=42,
    early_stopping=True,
    scoring='f1_macro'
)

# ── 6. Avaliación CV ─────────────────────────────────────────────────────────
# Pasamos os fit_params cos pesos para que os teña en conta ao adestrar
cv_scores = cross_val_score(
    model, X_train, y_train, cv=5, scoring='f1_macro', params={'sample_weight': pesos_mostras}
)
print(f"CV F1-Macro (HistGradientBoosting - All Features): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 7. Adestramento e Predición ──────────────────────────────────────────────
model.fit(X_train, y_train, sample_weight=pesos_mostras)

preds = model.predict(X_test)

submission = pd.DataFrame({
    'ID_Cliente': test['ID_Cliente'],
    'Target_Risco': preds
})
submission.to_csv('./resultados/submission_hgb_all.csv', index=False)
print(f"\nArquivo xerado: submission_hgb_all.csv ({len(submission)} filas)")