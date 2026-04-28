import pandas as pd
import numpy as np
from scipy.stats import mode as scipy_mode
import warnings

# Algoritmos
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
from sklearn.base import clone

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

from funciones.preprocesado2x2 import preprocesar_datos
from funciones.modelado2x2 import crear_catboost

# ── 0. Configuración e Constantes ─────────────────────────────────────────────
SEED      = 777
CV_FOLDS  = 5
CV_GAP    = 250
N_CLASSES = 4
RUTA_TRAIN = './data/train.csv'
RUTA_TEST  = './data/test.csv'

# AS TÚAS 16 VARIABLES DE OURO (15 Ratios + 1 Estacionalidade)
TOP_FEATURES = [
    'Historial_Impagos', 
    'Utilizacion_Credito', 
    'Consultas_Risco_6M', 
    'Codigo_Postal', 
    'Fondo_Emerxencia_Meses', 
    'ratio_debeda_ingresos',
    'Ratio_Cota_Ingresos', 
    'Indice_Estres_Financeiro', 
    'Anos_Emprego', 
    'Ingresos_Anuais', 
    'Variacion_Saldo_6M',
    'ratio_saldo_mensual',
    'Antiguedade_Cliente_Anos',
    'Saldo_Medio_3M',
    'Distancia_Oficina_Km',
    'Mes_Solicitude' # <--- A nosa nova arma temporal
]

# Hiperparámetros do meta-modelo (cos vosos pesos manuais)
META_PARAMS = dict(
    depth         = 6,
    iterations    = 1000, 
    learning_rate = 0.03,
    l2_leaf_reg   = 3.8,
    class_weights = [1.0, 2.1, 2.15, 3.8],
    loss_function = 'MultiClass',
    random_seed   = SEED,
    verbose       = 0,
    thread_count  = -1,
)

# ── 1. Carga ──────────────────────────────────────────────────────────────────
print("=" * 66)
print("  CARGA DE DATOS E ORDENACIÓN TEMPORAL")
print("=" * 66)

train_raw = pd.read_csv(RUTA_TRAIN)
test_raw  = pd.read_csv(RUTA_TEST)

ids_test     = test_raw['ID_Cliente'].copy()
# Ordenación temporal vital
train_raw    = train_raw.sort_values('Data_Solicitude').reset_index(drop=True)
dates_sorted = pd.to_datetime(train_raw['Data_Solicitude']).copy()

# Xa non borramos as columnas aquí. Deixamos a Data_Solicitude para que o 
# módulo de preprocesado poida extraer o Mes_Solicitude.
train_raw = train_raw.drop(columns=['ID_Cliente'])
test_raw  = test_raw.drop(columns=['ID_Cliente'])

print(f"  Train : {train_raw.shape[0]} filas · {train_raw.shape[1]} columnas")
print(f"  Test  : {test_raw.shape[0]} filas\n")

# ── 2. Preprocesado ───────────────────────────────────────────────────────────
print("=" * 66)
print("  PREPROCESADO DUAL (OHE vs NATIVO)")
print("=" * 66)

X_ohe, T_ohe, y_train, CATEGORICAS = preprocesar_datos(train_raw, test_raw, usar_ohe=True)
X_nat, T_nat, _, _ = preprocesar_datos(train_raw, test_raw, usar_ohe=False)

X_ohe   = X_ohe.reset_index(drop=True)
X_nat   = X_nat.reset_index(drop=True)
T_ohe   = T_ohe.reset_index(drop=True)
T_nat   = T_nat.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

n_train     = len(X_nat)
dates_train = dates_sorted.iloc[:n_train].reset_index(drop=True)

# Forzamos a que o Mes sexa tratado como categoría polo Meta-Modelo CatBoost
if 'Mes_Solicitude' in X_nat.columns:
    X_nat['Mes_Solicitude'] = X_nat['Mes_Solicitude'].astype(str)
    T_nat['Mes_Solicitude'] = T_nat['Mes_Solicitude'].astype(str)
    if 'Mes_Solicitude' not in CATEGORICAS:
        CATEGORICAS.append('Mes_Solicitude')

# Impoñemos as nosas 16 variables cirurxicamente
BEST_FEATS = [c for c in TOP_FEATURES if c in X_nat.columns]
cats_best  = [c for c in CATEGORICAS if c in BEST_FEATS]
cat_idx_best = [BEST_FEATS.index(c) for c in cats_best]

print(f"  Variables seleccionadas ({len(BEST_FEATS)}): {BEST_FEATS}")
print(f"  Categóricas activas: {cats_best}")

X_sel = X_nat[BEST_FEATS].copy()
T_sel = T_nat[BEST_FEATS].copy()

# ── 3. Modelos Base (O Comité de Sabios) ──────────────────────────────────────
print("\n" + "=" * 66)
print("  CONSTRUCIÓN DOS MODELOS BASE")
print("=" * 66)

base_models = []

# 1. Random Forest (Calibrado para probabilidades reais e precisas)
rf_base = RandomForestClassifier(n_estimators=400, max_depth=12, class_weight='balanced', random_state=SEED, n_jobs=-1)
rf_calibrado = CalibratedClassifierCV(rf_base, cv=3)
base_models.append(('rf_calibrado', rf_calibrado))

# 2. Regresión Loxística (Apoio lineal blindado contra Nulos)
lr_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED))
])
base_models.append(('loxistica', lr_pipeline))

# 3. LightGBM
if LGBM_OK:
    base_models.append(('lgbm', LGBMClassifier(n_estimators=600, learning_rate=0.05, max_depth=7, class_weight='balanced', random_state=SEED, n_jobs=-1, verbose=-1)))

# 4. XGBoost
if XGB_OK:
    base_models.append(('xgb', XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, eval_metric='mlogloss', early_stopping_rounds=40, random_state=SEED, n_jobs=-1, verbosity=0)))

# 5. CatBoost OHE
base_models.append(('catboost_ohe', CatBoostClassifier(iterations=800, learning_rate=0.04, depth=7, l2_leaf_reg=3.0, auto_class_weights='Balanced', random_seed=SEED, verbose=0, early_stopping_rounds=40)))

print(f"  Modelos base activos: {[n for n, _ in base_models]}\n")

# ── 4. Xeración de OOF Temporalmente Honesta ─────────────────────────────────
print("=" * 66)
print("  XERACIÓN DE OOF (TimeSeriesSplit)")
print("=" * 66)

tscv      = TimeSeriesSplit(n_splits=CV_FOLDS, gap=CV_GAP)
oof_probs = {name: np.zeros((n_train, N_CLASSES)) for name, _ in base_models}
oof_mask  = np.zeros(n_train, dtype=bool)

def _fit(name, model, Xtr, ytr, Xval=None, yval=None):
    if 'xgb' in name:
        sw = compute_sample_weight('balanced', ytr)
        if Xval is not None:
            model.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xval, yval)], verbose=False)
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

for fold, (idx_tr, idx_val) in enumerate(tscv.split(X_ohe), 1):
    Xtr_f, ytr_f   = X_ohe.iloc[idx_tr], y_train.iloc[idx_tr]
    Xval_f, yval_f = X_ohe.iloc[idx_val], y_train.iloc[idx_val]
    oof_mask[idx_val] = True
    
    d_max = dates_train.iloc[idx_tr].max().date()
    d_min = dates_train.iloc[idx_val].min().date()
    print(f"  Fold {fold} | Train ≤ {d_max} | Val ≥ {d_min} | Val Size: {len(idx_val)}")
    
    for name, model in base_models:
        _fit(name, model, Xtr_f, ytr_f, Xval_f, yval_f)
        probs = model.predict_proba(Xval_f)
        oof_probs[name][idx_val] = probs

# ── 5. Adestramento final dos modelos base ───────────────────────────────────
print("\n  Adestramento final dos modelos base sobre todo o histórico...")
test_probs = {}

def _clone_for_final(name, model):
    if 'xgb' in name:
        params = model.get_params()
        params['early_stopping_rounds'] = None
        return model.__class__(**params)
    elif 'catboost' in name:
        params = model.get_params()
        params.pop('early_stopping_rounds', None)
        return model.__class__(**params)
    else:
        return clone(model)

for name, model in base_models:
    final_model = _clone_for_final(name, model)
    _fit(name, final_model, X_ohe, y_train)
    test_probs[name] = final_model.predict_proba(T_ohe)
    print(f"    {name} finalizado.")

# ── 6. Construción das Meta-Features ─────────────────────────────────────────
print("\n" + "=" * 66)
print("  EXTRACCIÓN DE INCERTEZA E META-FEATURES")
print("=" * 66)

def _build_meta_df(probs_dict, mask=None):
    frames = []
    for name in probs_dict:
        p = probs_dict[name] if mask is None else probs_dict[name][mask]
        frames.append(pd.DataFrame(p, columns=[f'{name}_p{c}' for c in range(N_CLASSES)]))
    
    all_p = np.stack([probs_dict[n] if mask is None else probs_dict[n][mask] for n in probs_dict])
    
    for i, name in enumerate(probs_dict):
        p = all_p[i]
        entropy = -np.sum(p * np.log(np.clip(p, 1e-10, 1)), axis=1)
        frames.append(pd.DataFrame({f'{name}_entropy': entropy}))

    disagreement = all_p.var(axis=0).mean(axis=1)
    frames.append(pd.DataFrame({'model_disagreement': disagreement}))
    
    votes = all_p.argmax(axis=2)
    maj_vote = scipy_mode(votes, axis=0, keepdims=False).mode.ravel()
    frames.append(pd.DataFrame({'majority_vote': maj_vote.astype(str)}))
    return pd.concat(frames, axis=1).reset_index(drop=True)

oof_meta_df  = _build_meta_df(oof_probs, mask=oof_mask)
test_meta_df = _build_meta_df(test_probs, mask=None)

X_sel_masked = X_sel.reset_index(drop=True)[oof_mask].reset_index(drop=True)
y_meta       = y_train[oof_mask].reset_index(drop=True)

X_meta = pd.concat([X_sel_masked, oof_meta_df], axis=1)
T_meta = pd.concat([T_sel.reset_index(drop=True), test_meta_df], axis=1)

meta_cols    = X_meta.columns.tolist()
cat_idx_meta = cat_idx_best + [meta_cols.index('majority_vote')]

# ── 7. Adestramento do Meta-Modelo (Con Early Stopping Protegido) ────────────
print("\n" + "=" * 66)
print("  ADESTRAMENTO DO META-MODELO CATBOOST")
print("=" * 66)

# Facemos un split temporal do 90/10 sobre as meta-features para o early_stopping
split_idx = int(len(X_meta) * 0.9)
X_meta_tr, y_meta_tr   = X_meta.iloc[:split_idx], y_meta.iloc[:split_idx]
X_meta_val, y_meta_val = X_meta.iloc[split_idx:], y_meta.iloc[split_idx:]

meta_final = CatBoostClassifier(**META_PARAMS, early_stopping_rounds=50)

print(f"  Adestrando Meta-modelo sobre {len(X_meta_tr)} filas | Validando en {len(X_meta_val)} filas...")
meta_final.fit(
    Pool(X_meta_tr, label=y_meta_tr, cat_features=cat_idx_meta),
    eval_set=Pool(X_meta_val, label=y_meta_val, cat_features=cat_idx_meta)
)

print(f"  O Meta-modelo detívose na iteración: {meta_final.get_best_iteration()}")

# Mostrar as importancias finais
imp_meta = pd.Series(
    meta_final.get_feature_importance(Pool(X_meta, label=y_meta, cat_features=cat_idx_meta)),
    index=X_meta.columns
).sort_values(ascending=False)
print("\n  Top-10 meta-features por importancia:")
for i, (f, v) in enumerate(imp_meta.head(10).items(), 1):
    print(f"    {i:>2}. {f:<38}  {v:.4f}")

# ── 8. Predición e Exportación ───────────────────────────────────────────────
print("\n" + "=" * 66)
print("  PREDICIÓN OOT E EXPORTACIÓN")
print("=" * 66)

preds_test = meta_final.predict(Pool(T_meta, cat_features=cat_idx_meta)).ravel().astype(int)

submission = pd.DataFrame({'ID_Cliente': ids_test, 'Target_Risco': preds_test})
nome_arquivo = f'./resultados/28-04-2026_4.csv'
submission.to_csv(nome_arquivo, index=False)

print(f"  Arquivo xerado : {nome_arquivo}  ({len(submission)} filas)")
print("\n✔ Proceso mestre completado. Moita sorte no Leaderboard!")