import pandas as pd
import numpy as np

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

from funciones.preprocesado import seleccion_de_variables
from funciones.modelado import adestrar_por_stacking

import warnings
warnings.filterwarnings('ignore')

# ── 0. Variables globais ─────────────────────────────────────────────────────

SEED    = 42
N_MIN   = 15    # mínimo de variables a probar
N_MAX   = 20    # máximo de variables a probar
N_STEP  =  1    # paso do bucle
CV      =  5    # folds para a validación cruzada

# ── 1. Cargar datos ──────────────────────────────────────────────────────────
train = pd.read_csv('./data/train.csv')
test  = pd.read_csv('./data/test.csv')

# ── 2. Preprocesamento inicial (Codificación e limpeza) ──────────────────────
y_train = train['Target_Risco'].copy()

X_train_raw = train.drop(columns=['ID_Cliente', 'Target_Risco', 'Data_Solicitude'])
X_test_raw  = test.drop(columns=['ID_Cliente', 'Data_Solicitude'])

X_train_encoded = pd.get_dummies(X_train_raw)
X_test_encoded  = pd.get_dummies(X_test_raw)

X_train_encoded, X_test_encoded = X_train_encoded.align(
    X_test_encoded, join='left', axis=1, fill_value=0
)

# ── 3. Definición dos modelos base e do meta-modelo ──────────────────────────
#
#   Engadimos dous novos modelos respecto á versión anterior:
#
#   · ExtraTreesClassifier  → como o RF pero con divisións aleatorias; achega
#                             máis diversidade e adoita ser complementario.
#   · SVC (rbf, proba)      → marxes de separación no espazo de características;
#                             perspectiva xeométrica moi distinta dos árbores.
#
#   Mantemos os catro anteriores (RF, GB, MLP, KNN) por seguir sendo útiles.

modelos_base = [
    ('random_forest', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1,
    )),
    ('extra_trees', ExtraTreesClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1,
    )),
    ('gradient_boosting', GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=SEED,
    )),
    ('svc', SVC(
        kernel='rbf',
        probability=True,       # necesario para stack_method='predict_proba'
        class_weight='balanced',
        random_state=SEED,
    )),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=400,
        early_stopping=True,
        random_state=SEED,
    )),
    ('knn', KNeighborsClassifier(
        n_neighbors=7,
        n_jobs=-1,
    )),
]

meta_modelo = LogisticRegression(max_iter=1000, random_state=SEED)

# ── 4. Bucle de selección de variables ───────────────────────────────────────
#
#   Para cada n en [N_MIN, N_MAX] con paso N_STEP:
#     1. Seleccionamos as top-n variables por correlación.
#     2. Imputamos nulos.
#     3. Avaliamos o stacking con validación cruzada (sen adestrar en todo train
#        aínda, iso só se fai unha vez co mellor n).
#
#   Gardamos o mellor n e o seu score para o adestramento final.

resultados = []  # lista de (n, cv_mean, cv_std)

rango = range(N_MIN, N_MAX + 1, N_STEP)
total = len(list(rango))

print("=" * 55)
print(f"  BÚSQUEDA DE VARIABLES: n ∈ [{N_MIN}, {N_MAX}], paso={N_STEP}")
print("=" * 55 + "\n")

for i, n in enumerate(range(N_MIN, N_MAX + 1, N_STEP), start=1):
    print(f"[{i}/{total}] Probando con n={n} variables...")

    features = seleccion_de_variables(X_train_encoded, y_train, n)

    X_tr = X_train_encoded[features].copy()
    for col in features:
        X_tr[col] = X_tr[col].fillna(X_tr[col].median())

    # Creamos un StackingClassifier temporal só para a CV
    stk_tmp = StackingClassifier(
        estimators=modelos_base,
        final_estimator=meta_modelo,
        cv=CV,
        stack_method='predict_proba',
        n_jobs=-1,
        passthrough=False,
    )

    cv_scores = cross_val_score(stk_tmp, X_tr, y_train, cv=CV, scoring='f1_macro')
    print(f"  → CV F1-Macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

    resultados.append((n, cv_scores.mean(), cv_scores.std(), features))

# ── 5. Resumo e selección do mellor n ────────────────────────────────────────
print("\n" + "=" * 55)
print("  RESUMO DE RESULTADOS")
print("=" * 55)
print(f"  {'n':>4}   {'CV F1-Macro':>12}   {'±':>8}")
print("  " + "-" * 30)
for n, mean, std, _ in resultados:
    print(f"  {n:>4}   {mean:>12.4f}   {std:>8.4f}")

mellor = max(resultados, key=lambda x: x[1])
BEST_N, best_mean, best_std, BEST_FEATURES = mellor

print("\n" + "=" * 55)
print(f"  MELLOR: n={BEST_N}  →  CV F1-Macro={best_mean:.4f} ± {best_std:.4f}")
print("=" * 55 + "\n")

# ── 6. Adestramento final co mellor n ────────────────────────────────────────
X_train_best = X_train_encoded[BEST_FEATURES].copy()
X_test_best  = X_test_encoded[BEST_FEATURES].copy()

for col in BEST_FEATURES:
    median = X_train_best[col].median()
    X_train_best[col] = X_train_best[col].fillna(median)
    X_test_best[col]  = X_test_best[col].fillna(median)

print(f"Adestramento final con n={BEST_N} variables...")
stacking_clf, preds = adestrar_por_stacking(
    modelos_base=modelos_base,
    X_train=X_train_best,
    y_train=y_train,
    X_test=X_test_best,
    meta_modelo=meta_modelo,
    cv=CV,
)

# ── 7. Arquivo de envío ──────────────────────────────────────────────────────
nome_arquivo = f'./resultados/17-04-2026-stacking-features{BEST_N}.csv'

submission = pd.DataFrame({
    'ID_Cliente':   test['ID_Cliente'],
    'Target_Risco': preds,
})
submission.to_csv(nome_arquivo, index=False)
print(f"\nArquivo xerado con éxito: {nome_arquivo} ({len(submission)} filas)")