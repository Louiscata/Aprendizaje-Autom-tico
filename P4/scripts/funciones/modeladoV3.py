import pandas as pd
import numpy as np

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings('ignore')

def adestrar_por_stacking(
    modelos_base: list,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    meta_modelo=None,
    cv: int=5,
    seed: int=42
) -> tuple[object, np.ndarray]:
    """
    Adestra un conxunto de modelos base e un meta-modelo mediante stacking.

    O stacking funciona así:
      1. Cada modelo base adéstrase con validación cruzada sobre X_train e as
         súas predicións out-of-fold constrúen un novo dataset de probabilidades.
      2. O meta-modelo adéstrase sobre ese dataset de meta-features.
      3. Para predicir en test, cada modelo base predí sobre X_test e o
         meta-modelo combina esas predicións.
 
    Parámetros
    ----------
    modelos_base : Lista de tuplas (nome, estimador) compatibles con sklearn.
    X_train      : Features de adestramento.
    y_train      : Target de adestramento.
    X_test       : Features de test.
    meta_modelo  : Estimador para combinar as saídas dos modelos base.
                   Por defecto úsase LogisticRegression (NEUTRAL, sen class_weight).
    cv           : Número de folds para a validación cruzada interna.
 
    Devolve
    -------
    stacking_clf : O StackingClassifier xa adestrado.
    preds_test   : Array coas predicións finais sobre X_test.
    """
    
    if meta_modelo is None:
        meta_modelo = LogisticRegression(
            max_iter=1000, 
            C=0.1, 
            random_state=seed
        )
 
    # Particións balanceadas
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    stacking_clf = StackingClassifier(
        estimators=modelos_base,
        final_estimator=meta_modelo,
        cv=skf,
        stack_method='predict_proba',
        n_jobs=-1,
        passthrough=False,              
    )
 
    # ── Validación cruzada do pipeline completo ──────────────────────────────
    print("Executando validación cruzada do stacking")
    cv_scores = cross_val_score(stacking_clf, X_train, y_train, cv=skf, scoring='f1_macro')
    print(f"  CV F1-Macro (Stacking): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
 
    # ── Adestramento final sobre todos os datos de train ─────────────────────
    print("Adestramento final do stacking")
    stacking_clf.fit(X_train, y_train)
 
    preds_train = stacking_clf.predict(X_train)
    f1_train = f1_score(y_train, preds_train, average='macro')
    print(f"Train F1-Macro: {f1_train:.4f}")
    
    # Reporte por clase para verificar que non hai sobre-compensación
    f1_por_clase = f1_score(y_train, preds_train, average=None)
    for i, f1 in enumerate(f1_por_clase):
        print(f"    · Clase {i}: F1 = {f1:.4f}")
    print()
 
    # ── Predicións sobre test ─────────────────────────────────────────────────
    preds_test = stacking_clf.predict(X_test)
 
    return stacking_clf, preds_test

def adestrar_por_blending(
    modelos_base: list,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    meta_modelo=None,
    test_size: float=0.2,
    seed: int=42
) -> tuple[object, np.ndarray]:
    """
    Adestra un conxunto de modelos base e un meta-modelo mediante blending puro.
    
    Pasos:
      1. Divide o adestramento nun X_base (ex. 80%) e un X_holdout (ex. 20%).
      2. Adestra os modelos base ÚNICAMENTE no X_base.
      3. Extrae as probabilidades predict_proba dos modelos no X_holdout.
      4. Adestra o meta-modelo sobre as probabilidades do X_holdout.
      5. Predí sobre test usando as probabilidades que xeran os modelos base no X_test.
    """
    if meta_modelo is None:
        # Loxística neutral e regularizada para protexer contra sobreaxuste
        meta_modelo = LogisticRegression(max_iter=1000, C=0.1, random_state=seed)

    # ── Paso 1: Partición estrita (Holdout) ──────────────────────────────────
    X_base, X_holdout, y_base, y_holdout = train_test_split(
        X_train, y_train, 
        test_size=test_size, 
        random_state=seed, 
        stratify=y_train
    )

    meta_features_holdout = []
    meta_features_test = []

    # ── Paso 2: Adestrar bases e extraer probabilidades ──────────────────────
    print(f"Adestrando {len(modelos_base)} modelos base no {100-int(test_size*100)}% dos datos...")
    
    for nome, modelo in modelos_base:
        print(f"  -> Adestrando {nome}...")
        modelo.fit(X_base, y_base)
        
        # O segredo do éxito: pasar probabilidades, non predicións duras
        probs_holdout = modelo.predict_proba(X_holdout)
        probs_test    = modelo.predict_proba(X_test)
        
        meta_features_holdout.append(probs_holdout)
        meta_features_test.append(probs_test)

    # ── Paso 3: Construír as matrices do meta-modelo ─────────────────────────
    # Xuntamos todas as probabilidades (Ex: 3 modelos x 4 clases = 12 columnas)
    X_meta_holdout = np.hstack(meta_features_holdout)
    X_meta_test    = np.hstack(meta_features_test)

    # ── Paso 4: Adestrar e avaliar o meta-modelo ─────────────────────────────
    print(f"\nAdestrando o meta-modelo no {int(test_size*100)}% de Holdout...")
    meta_modelo.fit(X_meta_holdout, y_holdout)

    # Avaliamos no propio holdout para confirmar que aprendeu ben a mestura
    preds_holdout = meta_modelo.predict(X_meta_holdout)
    f1_holdout = f1_score(y_holdout, preds_holdout, average='macro')
    
    print(f"Holdout F1-Macro (Meta-modelo): {f1_holdout:.4f}")
    
    f1_por_clase = f1_score(y_holdout, preds_holdout, average=None)
    for i, f1 in enumerate(f1_por_clase):
        print(f"    · Clase {i}: F1 = {f1:.4f}")
    print()

    # ── Paso 5: Predicións finais sobre Kaggle ───────────────────────────────
    preds_test = meta_modelo.predict(X_meta_test)

    return meta_modelo, preds_test