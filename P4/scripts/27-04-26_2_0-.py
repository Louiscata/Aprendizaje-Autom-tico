"""
catboost_sum_native.py
──────────────────────
Competición Kaggle – Clasificación de risco crediticio (Target_Risco: 0-3)

Estratexia:
  1. Preprocesado con preprocesar_datos() sen OHE → categóricas nativas CatBoost
  2. Selección das mellores N variables (N en 11..17) por importancia CatBoost + CV
  3. Validación temporal con TimeSeriesSplit(gap=G):
       · train = rexistros máis antigos  |  val = rexistros máis recentes
       · o parámetro 'gap' exclúe G rexistros entre train e val para evitar
         que solicitudes moi próximas no tempo contaminen a validación
       · isto é máis realista que TimeSeriesSplit puro para proxectar ao futuro
  4. auto_class_weights='Balanced' en todos os modelos
  5. Adestramento de varios CatBoost (via crear_catboost) + combinación con sum_models
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))   # para importar os módulos locais

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool, sum_models
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

from funciones.preprocesado2x2 import preprocesar_datos
from funciones.modelado2x2 import crear_catboost

# ── 0. Configuración ─────────────────────────────────────────────────────────
SEED         = 777
CV_FOLDS     = 5
# Gap en nº de rexistros entre o último exemplo de train e o primeiro de val.
# Con ~18 000 rexistros e 3 anos de datos, ~250 filas ≈ 1 mes de marxe.
CV_GAP       = 250
N_MIN        = 11
N_MAX        = 17
RUTA_TRAIN   = './data/train.csv'
RUTA_TEST    = './data/test.csv'

# Modelos que se validarán E combinarán con sum_models.
# Diversidade por semente, profundidade, lr e l2 para que o ensemble gañe robustez.
CONFIGS = [
    dict(seed=777,  depth=7, iterations=1200, lr=0.03, l2=3.0,  balance='Balanced'),
    dict(seed=42,   depth=7, iterations=1200, lr=0.03, l2=3.0,  balance='Balanced'),
    dict(seed=123,  depth=8, iterations=1000, lr=0.04, l2=1.0,  balance='Balanced'),
    dict(seed=999,  depth=6, iterations=1000, lr=0.04, l2=5.0,  balance='Balanced'),
    dict(seed=2024, depth=7, iterations=1400, lr=0.02, l2=3.0,  balance='Balanced'),
]

# ── 1. Carga de datos ────────────────────────────────────────────────────────
print("=" * 62)
print("  CARGA DE DATOS")
print("=" * 62)
train_raw = pd.read_csv(RUTA_TRAIN)
test_raw  = pd.read_csv(RUTA_TEST)
print(f"  Train bruto : {train_raw.shape[0]} filas · {train_raw.shape[1]} columnas")
print(f"  Test  bruto : {test_raw.shape[0]} filas · {test_raw.shape[1]} columnas\n")

# Gardamos as datas ANTES do preprocesado para ordear e facer o split temporal.
# A data NON se usa como feature en ningún momento.
train_raw['Data_Solicitude'] = pd.to_datetime(train_raw['Data_Solicitude'])
test_raw['Data_Solicitude']  = pd.to_datetime(test_raw['Data_Solicitude'])
train_raw = train_raw.sort_values('Data_Solicitude').reset_index(drop=True)
ids_test  = test_raw['ID_Cliente'].copy()

print(f"  Rango temporal train : {train_raw['Data_Solicitude'].min().date()}  →  "
      f"{train_raw['Data_Solicitude'].max().date()}\n")

# ── 2. Preprocesado ──────────────────────────────────────────────────────────
print("=" * 62)
print("  PREPROCESADO  (sen OHE → categóricas nativas CatBoost)")
print("=" * 62)

X_all, X_test, y_train, CATEGORICAS = preprocesar_datos(
    train_raw, test_raw, usar_ohe=False
)

# Índices de columna das categóricas (necesarios para Pool de CatBoost)
cols_list  = X_all.columns.tolist()
cat_idx_all = [cols_list.index(c) for c in CATEGORICAS if c in cols_list]

# Gardar as datas aliñadas co índice do train despois de drop_duplicates
# (preprocesar_datos fai reset_index internamente; realiñamos pola orde)
dates_train = train_raw.loc[X_all.index, 'Data_Solicitude'].reset_index(drop=True)
X_all = X_all.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

print(f"\n  Categóricas que usará CatBoost de forma nativa : {CATEGORICAS}\n")

# ── 3. Selección do mellor N (11 → 17) ───────────────────────────────────────
# Paso 3a: modelo auxiliar sobre TODAS as variables para obter o ranking
print("=" * 62)
print(f"  SELECCIÓN DE VARIABLES  (N de {N_MIN} a {N_MAX})")
print("=" * 62)

print("\n  3a. Calculando importancias con modelo auxiliar...")
aux = crear_catboost(seed=SEED, depth=6, iterations=400, lr=0.08, l2=3.0,
                     balance='Balanced')
aux.fit(Pool(X_all, label=y_train, cat_features=cat_idx_all))

importancias = pd.Series(
    aux.get_feature_importance(Pool(X_all, label=y_train, cat_features=cat_idx_all)),
    index=X_all.columns
).sort_values(ascending=False)

print(f"\n  Ranking completo de variables:")
for i, (feat, imp) in enumerate(importancias.items(), 1):
    print(f"    {i:>2}. {feat:<38}  {imp:.4f}")

RANKING = importancias.index.tolist()

# Paso 3b: CV temporal para cada N, co mesmo TimeSeriesSplit(gap=CV_GAP)
# O gap evita que rexistros moi próximos no tempo entre train e val
# produzan un optimismo artificial na métrica.
print(f"\n  3b. CV temporal por N  (TimeSeriesSplit · folds={CV_FOLDS} · gap={CV_GAP})")
print(f"  {'N':>3}   F1-macro medio   std")
print(f"  {'-'*35}")

tscv = TimeSeriesSplit(n_splits=CV_FOLDS, gap=CV_GAP)
resultados_n = {}

for n in range(N_MIN, N_MAX + 1):
    feats_n  = RANKING[:n]
    cats_n   = [c for c in CATEGORICAS if c in feats_n]
    cat_idx_n = [feats_n.index(c) for c in cats_n]

    f1s = []
    for idx_tr, idx_val in tscv.split(X_all):
        Xtr, ytr   = X_all[feats_n].iloc[idx_tr],  y_train.iloc[idx_tr]
        Xval, yval = X_all[feats_n].iloc[idx_val], y_train.iloc[idx_val]

        m = crear_catboost(seed=SEED, depth=7, iterations=600, lr=0.05,
                           l2=3.0, balance='Balanced')
        m.fit(
            Pool(Xtr, label=ytr, cat_features=cat_idx_n),
            eval_set=Pool(Xval, label=yval, cat_features=cat_idx_n),
        )
        preds = m.predict(Xval).ravel().astype(int)
        f1s.append(f1_score(yval, preds, average='macro'))

    media, std = float(np.mean(f1s)), float(np.std(f1s))
    resultados_n[n] = media
    print(f"  N={n:>2}   {media:.5f}          {std:.5f}")

BEST_N = max(resultados_n, key=resultados_n.get)
BEST_FEATS = RANKING[:BEST_N]
cats_best  = [c for c in CATEGORICAS if c in BEST_FEATS]
cat_idx_best = [BEST_FEATS.index(c) for c in cats_best]

print(f"\n  ✔  Mellor N = {BEST_N}  →  F1-macro CV = {resultados_n[BEST_N]:.5f}")
print(f"  Variables seleccionadas:")
for i, f in enumerate(BEST_FEATS, 1):
    print(f"    {i:>2}. {f}")

X = X_all[BEST_FEATS].copy()
T = X_test[BEST_FEATS].copy()

# ── 4. Validación temporal + adestramento final de cada modelo ────────────────
# Para cada config en CONFIGS:
#   a) CV temporal con TimeSeriesSplit(gap=CV_GAP) → F1-macro real
#   b) Adestramento final sobre todo o train co mesmo config
# O que se valida e o que se combina son exactamente os mesmos modelos.
print("\n" + "=" * 62)
print(f"  VALIDACIÓN + ADESTRAMENTO FINAL")
print(f"  ({len(CONFIGS)} configs · TimeSeriesSplit · folds={CV_FOLDS} · gap={CV_GAP})")
print("=" * 62)

full_pool   = Pool(X, label=y_train, cat_features=cat_idx_best)
modelos     = []
f1_configs  = []

for i, cfg in enumerate(CONFIGS, 1):
    desc = (f"seed={cfg['seed']}, depth={cfg['depth']}, "
            f"iter={cfg['iterations']}, lr={cfg['lr']}, l2={cfg['l2']}")
    print(f"\n  ── Modelo {i}/{len(CONFIGS)}  ({desc})")

    # a) Validación temporal
    f1s = []
    for fold, (idx_tr, idx_val) in enumerate(tscv.split(X), 1):
        Xtr, ytr   = X.iloc[idx_tr],  y_train.iloc[idx_tr]
        Xval, yval = X.iloc[idx_val], y_train.iloc[idx_val]

        d_tr_max  = dates_train.iloc[idx_tr].max().date()
        d_val_min = dates_train.iloc[idx_val].min().date()

        m_cv = crear_catboost(**cfg)
        m_cv.fit(
            Pool(Xtr, label=ytr, cat_features=cat_idx_best),
            eval_set=Pool(Xval, label=yval, cat_features=cat_idx_best),
        )
        preds = m_cv.predict(Xval).ravel().astype(int)
        f1    = f1_score(yval, preds, average='macro')
        f1s.append(f1)
        print(f"    Fold {fold}  |  train ≤ {d_tr_max}  |  "
              f"val ≥ {d_val_min}  |  F1 = {f1:.5f}")

    f1_medio = float(np.mean(f1s))
    f1_configs.append(f1_medio)
    print(f"    → F1-macro medio : {f1_medio:.5f}  (std: {np.std(f1s):.5f})")

    # b) Adestramento final (mesmo config, sen early stopping, train completo)
    print(f"    Adestramento final...", end=' ', flush=True)
    # Para o adestramento final eliminamos early_stopping para usar tódalas iteracións
    m_final = CatBoostClassifier(
        iterations=cfg['iterations'],
        learning_rate=cfg['lr'],
        depth=cfg['depth'],
        l2_leaf_reg=cfg['l2'],
        auto_class_weights=cfg['balance'],
        random_seed=cfg['seed'],
        verbose=0,
    )
    m_final.fit(full_pool)
    modelos.append(m_final)
    print("listo.")

# Resumo
print(f"\n{'─' * 62}")
print("  RESUMO DE VALIDACIÓN:")
for i, (cfg, f1) in enumerate(zip(CONFIGS, f1_configs), 1):
    print(f"    Modelo {i}  (seed={cfg['seed']}, depth={cfg['depth']}, "
          f"lr={cfg['lr']})  →  {f1:.5f}")
print(f"    Media xeral : {np.mean(f1_configs):.5f}")
print(f"{'─' * 62}\n")

# ── 5. Combinación con sum_models ────────────────────────────────────────────
print("=" * 62)
print(f"  COMBINACIÓN  (sum_models · {len(modelos)} modelos · pesos uniformes)")
print("=" * 62)

pesos        = [1.0 / len(modelos)] * len(modelos)
modelo_final = sum_models(modelos, weights=pesos)
print("  Combinación completada.\n")

# ── 6. Predición e arquivo de envío ──────────────────────────────────────────
print("=" * 62)
print("  PREDICIÓN E EXPORTACIÓN")
print("=" * 62)

preds_test = modelo_final.predict(
    Pool(T, cat_features=cat_idx_best),
    prediction_type='Probability'
).argmax(axis=1).astype(int)

submission = pd.DataFrame({
    'ID_Cliente':   ids_test,
    'Target_Risco': preds_test,
})

nome_arquivo = f'./resultados/27-04-2026_2_catboost_sum_features{BEST_N}.csv'
submission.to_csv(nome_arquivo, index=False)

print(f"  Arquivo xerado : {nome_arquivo}  ({len(submission)} filas)")
print(f"  Distribución de clases preditas:")
print(submission['Target_Risco'].value_counts().sort_index().to_string(header=False))
print("\n✔ Proceso completado con éxito.")