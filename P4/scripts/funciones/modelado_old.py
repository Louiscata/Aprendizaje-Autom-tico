import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
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
    seed=42
) -> tuple[object, np.ndarray]:
    """
    Adestra un conxunto de modelos base e un meta-modelo mediante stacking.

    O stacking funciona así:
      1. Cada modelo base adéstrase con validación cruzada sobre X_train e as
         súas predicións out-of-fold constrúen un novo dataset (meta-features).
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
                   Por defecto úsase LogisticRegression.
    cv           : Número de folds para a validación cruzada interna.
 
    Devolve
    -------
    stacking_clf : O StackingClassifier xa adestrado.
    preds_test   : Array coas predicións finais sobre X_test.
    """
    if meta_modelo is None:
        meta_modelo = LogisticRegression(max_iter=1000, random_state=seed)
 
    stacking_clf = StackingClassifier(
        estimators=modelos_base,
        final_estimator=meta_modelo,
        cv=cv,
        stack_method='predict_proba',   # usa probabilidades como meta-features
        n_jobs=-1,
        passthrough=False,              # só pasan as saídas dos base, non X orixinal
    )
 
    # ── Validación cruzada do pipeline completo ──────────────────────────────
    print("Executando validación cruzada do stacking")
    cv_scores = cross_val_score(stacking_clf, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"CV F1-Macro (Stacking): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
 
    # ── Adestramento final sobre todos os datos de train ─────────────────────
    print("Adestramento final do stacking")
    stacking_clf.fit(X_train, y_train)
 
    preds_train = stacking_clf.predict(X_train)
    f1_train = f1_score(y_train, preds_train, average='macro')
    print(f"Train F1-Macro: {f1_train:.4f}\n")
 
    # ── Predicións sobre test ─────────────────────────────────────────────────
    preds_test = stacking_clf.predict(X_test)
 
    return stacking_clf, preds_test

    from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

def adestrar_por_bagging(
    modelo_base,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_estimators: int = 100,
    max_samples: float = 1.0,
    max_features: float = 1.0,
    cv: int = 5,
    seed: int = 42
) -> tuple[object, np.ndarray]:
    """
    Adestra un clasificador mediante bagging sobre un modelo base.

    O bagging funciona así:
      1. Créanse N subconxuntos do dataset de adestramento mediante mostraxe
         con reposición (bootstrap).
      2. Adéstrase unha copia do modelo base en cada subconxunto.
      3. A predición final obtense por votación maioritaria entre todos os
         modelos adestrados.

    Parámetros
    ----------
    modelo_base   : Estimador sklearn que se usará como base do bagging.
    X_train       : Features de adestramento.
    y_train       : Target de adestramento.
    X_test        : Features de test.
    n_estimators  : Número de modelos base a adestrar.
    max_samples   : Fracción (ou número) de mostras para cada modelo base.
    max_features  : Fracción (ou número) de features para cada modelo base.
    cv            : Número de folds para a validación cruzada.
    seed          : Semente para reproducibilidade.

    Devolve
    -------
    bagging_clf : O BaggingClassifier xa adestrado.
    preds_test  : Array coas predicións finais sobre X_test.
    """
    bagging_clf = BaggingClassifier(
        estimator=modelo_base,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=True,         # mostraxe con reposición (bootstrap clásico)
        bootstrap_features=False,
        n_jobs=-1,
        random_state=seed
    )

    # ── Validación cruzada ───────────────────────────────────────────────────
    print("Executando validación cruzada do bagging")
    cv_scores = cross_val_score(bagging_clf, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"CV F1-Macro (Bagging): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

    # ── Adestramento final ───────────────────────────────────────────────────
    print("Adestramento final do bagging")
    bagging_clf.fit(X_train, y_train)

    preds_train = bagging_clf.predict(X_train)
    f1_train = f1_score(y_train, preds_train, average='macro')
    print(f"Train F1-Macro: {f1_train:.4f}\n")

    # ── Predicións sobre test ────────────────────────────────────────────────
    preds_test = bagging_clf.predict(X_test)

    return bagging_clf, preds_test


def adestrar_por_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    tipo: str = 'gradient',
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    cv: int = 5,
    seed: int = 42
) -> tuple[object, np.ndarray]:
    """
    Adestra un clasificador mediante boosting.

    O boosting funciona así:
      1. Adéstrase un modelo débil inicial sobre os datos.
      2. Cada modelo seguinte ponse a foco nos exemplos que o anterior
         clasificou mal, incrementando o seu peso (AdaBoost) ou axustando
         os residuos do erro (Gradient Boosting).
      3. A predición final é unha combinación ponderada de todos os modelos.

    Parámetros
    ----------
    X_train       : Features de adestramento.
    y_train       : Target de adestramento.
    X_test        : Features de test.
    tipo          : Algoritmo de boosting: 'gradient' ou 'adaboost'.
    n_estimators  : Número de estimadores (rondas de boosting).
    learning_rate : Taxa de aprendizaxe; controla a contribución de cada árbol.
                    Valores pequenos require máis estimadores.
    max_depth     : Profundidade máxima de cada árbol base (só 'gradient').
    cv            : Número de folds para a validación cruzada.
    seed          : Semente para reproducibilidade.

    Devolve
    -------
    boosting_clf : O clasificador de boosting xa adestrado.
    preds_test   : Array coas predicións finais sobre X_test.
    """
    if tipo == 'gradient':
        boosting_clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=seed
        )
    elif tipo == 'adaboost':
        boosting_clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),  # stump clásico
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=seed,
            algorithm='SAMME'
        )
    else:
        raise ValueError(f"Tipo de boosting non recoñecido: '{tipo}'. Usa 'gradient' ou 'adaboost'.")

    # ── Validación cruzada ───────────────────────────────────────────────────
    print(f"Executando validación cruzada do boosting ({tipo})")
    cv_scores = cross_val_score(boosting_clf, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"CV F1-Macro (Boosting/{tipo}): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

    # ── Adestramento final ───────────────────────────────────────────────────
    print(f"Adestramento final do boosting ({tipo})")
    boosting_clf.fit(X_train, y_train)

    preds_train = boosting_clf.predict(X_train)
    f1_train = f1_score(y_train, preds_train, average='macro')
    print(f"Train F1-Macro: {f1_train:.4f}\n")

    # ── Predicións sobre test ────────────────────────────────────────────────
    preds_test = boosting_clf.predict(X_test)

    return boosting_clf, preds_test

def adestrar_por_blending(
    modelos_base: list,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    meta_modelo=None,
    test_size=0.2,
    seed=42
) -> tuple[object, np.ndarray]:
    """
    Adestra un conxunto de modelos base e un meta-modelo mediante blending.
    
    Pasos:
      1. Divide o adestramento nun base_train_data e un holdout_set.
      2. Adestra os modelos base no base_train_data e xera predicións.
      3. Adestra o meta-modelo coas variables orixinais + as predicións base.
      4. Predí sobre test usando as variables orixinais + as predicións base.
      
    Parámetros
    ----------
    modelos_base : Lista de tuplas (nome, estimador) compatibles con sklearn.
    X_train      : Features de adestramento.
    y_train      : Target de adestramento.
    X_test       : Features de test.
    meta_modelo  : Estimador para combinar as saídas dos modelos base.
                   Por defecto úsase LogisticRegression.
    test_size    : Proporción do conxunto de adestramento usada para o holdout.
    """
    if meta_modelo is None:
        meta_modelo = LogisticRegression(random_state=seed, class_weight='balanced')

    # ── Paso 1: Dividir o conxunto de adestramento ───────────────────────────
    X_base, X_holdout, y_base, y_holdout = train_test_split(
        X_train, y_train, 
        test_size=test_size, 
        random_state=seed, 
        stratify=y_train # Manter balance de clases (Revisar como funciona con ou sen el)
    )

    # Matrices para almacenar as predicións 
    meta_features_holdout = np.zeros((X_holdout.shape[0], len(modelos_base)))
    meta_features_test = np.zeros((X_test.shape[0], len(modelos_base)))

    # ── Paso 2: Adestrar modelos base e predicir no holdout/test ─────────────
    for i, (nome, modelo) in enumerate(modelos_base):
        print(f"   -> Adestrando co modelo {nome}")
        modelo.fit(X_base, y_base) # Adestrar co base
        meta_features_holdout[:, i] = modelo.predict(X_holdout) # Gardar as predicións do holdout
        meta_features_test[:, i] = modelo.predict(X_test)

    # ── Paso 3: Adestrar o meta-modelo (Atributos orixinais + Predicións) ────
    
    # Liadas de pandas e numpy
    X_holdout_orixinal = X_holdout.values if isinstance(X_holdout, pd.DataFrame) else X_holdout
    X_test_orixinal = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

    # Combinamos os atributos orixinais coas novas predicións (meta-features)
    X_holdout_meta = np.hstack((X_holdout_orixinal, meta_features_holdout))
    X_test_meta = np.hstack((X_test_orixinal, meta_features_test))

    # Adestramos o meta-modelo
    meta_modelo.fit(X_holdout_meta, y_holdout)

    # Avaliación rápida do meta-modelo no propio holdout para ver como de ben aprendeu
    preds_holdout_meta = meta_modelo.predict(X_holdout_meta)
    f1_holdout = f1_score(y_holdout, preds_holdout_meta, average='macro')
    print(f"   -> F1-Macro do meta-modelo no Holdout: {f1_holdout:.4f}")

    # ── Paso 4: Predicións finais ────────────────────────────────────────────

    preds_test = meta_modelo.predict(X_test_meta)

    return meta_modelo, preds_test