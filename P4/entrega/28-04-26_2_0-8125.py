import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from scipy.stats import mode as scipy_mode
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

try:
    from lightgbm import LGBMClassifier
    LGBM_OK = True
except ImportError:
    LGBM_OK = False
    print("[AVISO] LightGBM non dispoñible.")

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except ImportError:
    XGB_OK = False
    print("[AVISO] XGBoost non dispoñible.")

from preprocesado import preprocesar_datos
from modelado import crear_catboost

SEED      = 777
CV_FOLDS  = 5
CV_GAP    = 250   # filas de marxe entre train e val para evitar contaminación temporal
N_VALUES  = [12]  # número de features nativas para o meta-modelo
N_CLASSES = 4
RUTA_TRAIN = './data/train.csv'
RUTA_TEST  = './data/test.csv'

# Hiperparámetros do meta-modelo CatBoost
META_PARAMS = dict(
    depth         = 6,
    iterations    = 1000,
    learning_rate = 0.03,
    l2_leaf_reg   = 3.8,
    # Clases minoritarias pesan máis para compensar o desbalance
    class_weights = [1.0, 2.1, 2.15, 3.8],
    loss_function = 'MultiClass',
    random_seed   = SEED,
    verbose       = 100,
    thread_count  = -1,
)

print("=" * 66)
print("  CARGA DE DATOS")
print("=" * 66)

train_raw = pd.read_csv(RUTA_TRAIN)
test_raw  = pd.read_csv(RUTA_TEST)

ids_test     = test_raw['ID_Cliente'].copy()
# Ordenamos antes de nada para que o split temporal sexa correcto
train_raw    = train_raw.sort_values('Data_Solicitude').reset_index(drop=True)
dates_sorted = pd.to_datetime(train_raw['Data_Solicitude']).copy()

train_raw = train_raw.drop(columns=['ID_Cliente', 'Data_Solicitude'])
test_raw  = test_raw.drop(columns=['ID_Cliente', 'Data_Solicitude'])

print(f"  Train : {train_raw.shape[0]} filas · {train_raw.shape[1]} columnas")
print(f"  Test  : {test_raw.shape[0]} filas\n")

print("=" * 66)
print("  PREPROCESADO")
print("=" * 66)

# Con OHE para os modelos base que non soportan categóricas nativas (RF, LGBM, XGB)
print("\n  [Con OHE] Para modelos base:")
X_ohe, T_ohe, y_train, CATEGORICAS = preprocesar_datos(
    train_raw, test_raw, usar_ohe=True
)

# Sen OHE para o meta-modelo CatBoost, que as xestiona mellor de forma nativa
print("\n  [Sen OHE] Para features orixinais do meta-modelo:")
X_nat, T_nat, _, _ = preprocesar_datos(
    train_raw, test_raw, usar_ohe=False
)

X_ohe   = X_ohe.reset_index(drop=True)
X_nat   = X_nat.reset_index(drop=True)
T_ohe   = T_ohe.reset_index(drop=True)
T_nat   = T_nat.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

n_train     = len(X_nat)
dates_train = dates_sorted.iloc[:n_train].reset_index(drop=True)

print("\n" + "=" * 66)
print(f"  SELECCIÓN DE VARIABLES NATIVAS  (N en {N_VALUES})")
print("=" * 66)

# Dia_Solicitude descartado: importancia case 0, non distingue o risco entre días
EXCLUIR_FEATURES = ['Dia_Solicitude']
cols_nat_candidatas = [c for c in X_nat.columns if c not in EXCLUIR_FEATURES]
X_nat_cand   = X_nat[cols_nat_candidatas]
T_nat_cand   = T_nat[cols_nat_candidatas]
cols_cand    = X_nat_cand.columns.tolist()
cat_idx_cand = [cols_cand.index(c) for c in CATEGORICAS if c in cols_cand]

print(f"\n  Excluídas de candidatas: {EXCLUIR_FEATURES}")
print("  Calculando importancias (modelo auxiliar CatBoost)...")

# Modelo rápido só para obter o ranking de importancias, non se usa para predicir
aux = crear_catboost(seed=SEED, depth=6, iterations=400, lr=0.08, l2=3.0,
                     balance='Balanced')
aux.fit(Pool(X_nat_cand, label=y_train, cat_features=cat_idx_cand))

importancias = pd.Series(
    aux.get_feature_importance(Pool(X_nat_cand, label=y_train,
                                    cat_features=cat_idx_cand)),
    index=X_nat_cand.columns
).sort_values(ascending=False)

print("\n  Ranking completo de variables candidatas:")
for i, (f, v) in enumerate(importancias.items(), 1):
    print(f"    {i:>2}. {f:<38}  {v:.4f}")

RANKING_NAT = importancias.index.tolist()

tscv = TimeSeriesSplit(n_splits=CV_FOLDS, gap=CV_GAP)
print(f"\n  CV temporal por N  (folds={CV_FOLDS}, gap={CV_GAP}):")

# N fixado en 12 tras experimentos
BEST_N = N_VALUES[0]
BEST_FEATS = RANKING_NAT[:BEST_N]
cats_best    = [c for c in CATEGORICAS if c in BEST_FEATS]
cat_idx_best = [BEST_FEATS.index(c) for c in cats_best]

print(f"  Features: {BEST_FEATS}\n")

X_sel = X_nat_cand[BEST_FEATS].copy()
T_sel = T_nat_cand[BEST_FEATS].copy()

print("=" * 66)
print("  MODELOS BASE")
print("=" * 66)

def _fit(name, model, Xtr, ytr, Xval=None, yval=None):
    # XGB necesita sample_weight explícito, CatBoost usa Pool o resto fit normal
    if 'xgb' in name:
        sw = compute_sample_weight('balanced', ytr)
        if Xval is not None:
            model.fit(Xtr, ytr, sample_weight=sw,
                      eval_set=[(Xval, yval)], verbose=False)
        else:
            model.fit(Xtr, ytr, sample_weight=sw)
    elif 'catboost' in name:
        pool_tr = Pool(Xtr, label=ytr)
        if Xval is not None:
            model.fit(pool_tr, eval_set=Pool(Xval, label=yval))
        else:
            model.fit(pool_tr)
    else:
        model.fit(Xtr, ytr)
    return model

def _clone_for_final(name, model):
    # No adestramento final non hai val set, así que eliminamos early_stopping
    import copy
    if 'xgb' in name:
        params = model.get_params()
        params['early_stopping_rounds'] = None
        return model.__class__(**params)
    elif 'catboost' in name:
        params = model.get_params()
        params.pop('early_stopping_rounds', None)
        return model.__class__(**params)
    else:
        return copy.deepcopy(model)

base_models = [
    ('random_forest', RandomForestClassifier(
        n_estimators=400, max_depth=12, class_weight='balanced',
        random_state=SEED, n_jobs=-1
    )),
]
if LGBM_OK:
    base_models.append(('lgbm', LGBMClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=7,
        class_weight='balanced', random_state=SEED, n_jobs=-1, verbose=-1
    )))
if XGB_OK:
    base_models.append(('xgb', XGBClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=6,
        eval_metric='mlogloss', early_stopping_rounds=40,
        random_state=SEED, n_jobs=-1, verbosity=0
    )))
base_models.append(('catboost_ohe', CatBoostClassifier(
    iterations=800, learning_rate=0.04, depth=7, l2_leaf_reg=3.0,
    auto_class_weights='Balanced', random_seed=SEED, verbose=0,
    early_stopping_rounds=40
)))

print(f"  Modelos base activos: {[n for n, _ in base_models]}\n")

print("=" * 66)
print("  XERACIÓN DE OOF  (split temporal · train=pasado · val=futuro)")
print("=" * 66)

# OOF: cada fila do train é predita por modelos que non a viron
tscv      = TimeSeriesSplit(n_splits=CV_FOLDS, gap=CV_GAP)
oof_probs = {name: np.zeros((n_train, N_CLASSES)) for name, _ in base_models}
oof_mask  = np.zeros(n_train, dtype=bool)  # marks filas que teñen predición OOF

for fold, (idx_tr, idx_val) in enumerate(tscv.split(X_ohe), 1):
    d_max = dates_train.iloc[idx_tr].max().date()
    d_min = dates_train.iloc[idx_val].min().date()
    print(f"\n  Fold {fold}  |  train ≤ {d_max}  |  val ≥ {d_min}"
          f"  |  val_size={len(idx_val)}")

    Xtr_f, ytr_f   = X_ohe.iloc[idx_tr], y_train.iloc[idx_tr]
    Xval_f, yval_f = X_ohe.iloc[idx_val], y_train.iloc[idx_val]
    oof_mask[idx_val] = True

    for name, model in base_models:
        _fit(name, model, Xtr_f, ytr_f, Xval_f, yval_f)
        probs = model.predict_proba(Xval_f)
        oof_probs[name][idx_val] = probs
        f1 = f1_score(yval_f, probs.argmax(axis=1), average='macro')
        print(f"    {name:<20} F1-val = {f1:.5f}")

n_oof  = oof_mask.sum()
n_excl = n_train - n_oof
# As filas máis antigas nunca son val, polo que quedan sen OOF e non se usan no meta-modelo
print(f"\n  Filas con OOF real : {n_oof}  |  "
      f"Excluídas : {n_excl} ({100*n_excl/n_train:.1f}%, datos máis antigos)")

print("\n  OOF global (só filas cubertas):")
for name in oof_probs:
    f1_oof = f1_score(y_train[oof_mask],
                      oof_probs[name][oof_mask].argmax(axis=1), average='macro')
    print(f"    {name:<20} F1-OOF = {f1_oof:.5f}")

# Reentrenamos cada modelo base con todo o train para xerar as predicions sobre test
print("\n  Adestramento final dos modelos base (train completo)...")
test_probs = {}
for name, model in base_models:
    final_model = _clone_for_final(name, model)
    _fit(name, final_model, X_ohe, y_train)
    test_probs[name] = final_model.predict_proba(T_ohe)
    print(f"    {name} listo.")

print("\n" + "=" * 66)
print("  CONSTRUCCIÓN DAS META-FEATURES + FEATURES DE INCERTIDUME")
print("=" * 66)

def _build_meta_df(probs_dict, mask=None):
    frames = []
    # Probabilidades de cada clase para cada modelo base
    for name in probs_dict:
        p = probs_dict[name] if mask is None else probs_dict[name][mask]
        frames.append(pd.DataFrame(
            p, columns=[f'{name}_p{c}' for c in range(N_CLASSES)]
        ))
    all_p = np.stack(
        [probs_dict[n] if mask is None else probs_dict[n][mask]
         for n in probs_dict]
    )  # forma: (n_models, n_rows, N_CLASSES)

    # Entropía por modelo: canto de inseguro está cada un na súa predición
    for i, name in enumerate(probs_dict):
        p = all_p[i]
        entropy = -np.sum(p * np.log(np.clip(p, 1e-10, 1)), axis=1)
        frames.append(pd.DataFrame({f'{name}_entropy': entropy}))

    # Desacordo entre modelos: varianza das probabilidades entre eles
    disagreement = all_p.var(axis=0).mean(axis=1)
    frames.append(pd.DataFrame({'model_disagreement': disagreement}))

    # Voto maioritario como feature categórica para que CatBoost o aproveite
    votes    = all_p.argmax(axis=2)
    maj_vote = scipy_mode(votes, axis=0, keepdims=False).mode.ravel()
    frames.append(pd.DataFrame({'majority_vote': maj_vote.astype(str)}))

    return pd.concat(frames, axis=1).reset_index(drop=True)

oof_meta_df  = _build_meta_df(oof_probs, mask=oof_mask)
test_meta_df = _build_meta_df(test_probs, mask=None)

X_sel_masked = X_sel.reset_index(drop=True)[oof_mask].reset_index(drop=True)
y_meta       = y_train[oof_mask].reset_index(drop=True)
dates_meta   = dates_train[oof_mask].reset_index(drop=True)

# Meta-features finais: top features nativas + saídas OOF + métricas de incertidume
X_meta = pd.concat([X_sel_masked, oof_meta_df], axis=1)
T_meta = pd.concat([T_sel.reset_index(drop=True), test_meta_df], axis=1)

meta_cols    = X_meta.columns.tolist()
cat_idx_meta = cat_idx_best + [meta_cols.index('majority_vote')]

print(f"  Dimensión meta-features train : {X_meta.shape}  "
      f"(excluídas {n_excl} filas sen OOF)")
print(f"  Dimensión meta-features test  : {T_meta.shape}")
print(f"  Cols de incertidume : entropía×{len(base_models)}, "
      f"desacordo, voto_maioritario\n")

print("=" * 66)
print("  ADESTRAMENTO DO META-MODELO (hiperparámetros hardcodeados)")
print("=" * 66)
print(f"\n  Params: depth={META_PARAMS['depth']}, "
      f"iter={META_PARAMS['iterations']}, "
      f"lr={META_PARAMS['learning_rate']:.6f}, "
      f"l2={META_PARAMS['l2_leaf_reg']:.4f}")
print(f"  Pesos de clase: {[round(w, 4) for w in META_PARAMS['class_weights']]}\n")

meta_final = CatBoostClassifier(**META_PARAMS)
meta_final.fit(Pool(X_meta, label=y_meta, cat_features=cat_idx_meta))

imp_meta = pd.Series(
    meta_final.get_feature_importance(
        Pool(X_meta, label=y_meta, cat_features=cat_idx_meta)
    ),
    index=X_meta.columns
).sort_values(ascending=False)

print("\n  Top-15 meta-features por importancia:")
for i, (f, v) in enumerate(imp_meta.head(15).items(), 1):
    print(f"    {i:>2}. {f:<38}  {v:.4f}")

print("\n" + "=" * 66)
print("  PREDICIÓN E EXPORTACIÓN")
print("=" * 66)

preds_train = meta_final.predict(
    Pool(X_meta, cat_features=cat_idx_meta)
).ravel().astype(int)

f1_train = f1_score(y_meta, preds_train, average='macro')
print(f"  F1 macro sobre train (meta, {len(y_meta)} filas con OOF): {f1_train:.5f}\n")

preds_test = meta_final.predict(
    Pool(T_meta, cat_features=cat_idx_meta)
).ravel().astype(int)

submission = pd.DataFrame({
    'ID_Cliente':   ids_test,
    'Target_Risco': preds_test,
})

nome_arquivo = f'./resultados/28-04-2026_2_catboost-stacking_final_top{len(BEST_FEATS)}.csv'
submission.to_csv(nome_arquivo, index=False)

print(f"  Arquivo xerado : {nome_arquivo}  ({len(submission)} filas)")
print(f"  Distribución de clases preditas:")
print(submission['Target_Risco'].value_counts().sort_index().to_string(header=False))
print("\n✔ Proceso completado con éxito.")