import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier, AdaBoostClassifier, BaggingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings('ignore')


def adestrar_por_stacking(
    modelos_base: list,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    meta_modelo=None,
    cv: int = 5,
    passthrough: bool = True,
    seed: int = 42,
) -> tuple[object, np.ndarray]:
    """
    Adestra un conxunto de modelos base e un meta-modelo mediante stacking.

    O stacking funciona así:
      1. Cada modelo base adéstrase con validación cruzada sobre X_train e as
         súas predicións out-of-fold constrúen un novo dataset (meta-features).
      2. O meta-modelo adéstrase sobre ese dataset de meta-features.
      3. Para predicir en test, cada modelo base predí sobre X_test e o
         meta-modelo combina esas predicións.

    Melloras respecto á versión anterior
    -------------------------------------
    · passthrough=True por defecto: o meta-modelo recibe tanto as saídas dos
      modelos base coma as features orixinais, o que lle permite correxir
      erros sistemáticos dos modelos base.
    · Meta-modelo con class_weight='balanced': evita que ignore as clases
      minoritarias (especialmente a clase 3, con só o 5% dos datos).
    · StratifiedKFold explícito no CV: garante que cada fold ten representación
      proporcional de todas as clases. Crítico con targets moi desbalanceados.
    · Reporte por clase no CV final: mostra o F1 de cada clase para detectar
      se o modelo segue ignorando as clases minoritarias.

    Parámetros
    ----------
    modelos_base : Lista de tuplas (nome, estimador) compatibles con sklearn.
    X_train      : Features de adestramento.
    y_train      : Target de adestramento.
    X_test       : Features de test.
    meta_modelo  : Estimador para combinar as saídas dos modelos base.
                   Por defecto: LogisticRegression con class_weight='balanced'.
    cv           : Número de folds para a validación cruzada interna.
    passthrough  : Se True, o meta-modelo recibe tamén as features orixinais.
                   Recomendado: True.
    seed         : Semente para reproducibilidade.

    Devolve
    -------
    stacking_clf : O StackingClassifier xa adestrado.
    preds_test   : Array coas predicións finais sobre X_test.
    """
    # [FIX] Meta-modelo con class_weight='balanced' para non ignorar clases
    # minoritarias. A versión anterior usaba LogisticRegression sen este
    # parámetro, o que penalizaba sistematicamente as clases 2 e 3.
    if meta_modelo is None:
        meta_modelo = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=seed,
        )

    # [FIX] passthrough=True: o meta-modelo ve as features orixinais ademais
    # das predicións dos modelos base. Permite correxir casos onde todos os
    # base models se equivocan no mesmo sentido.
    stacking_clf = StackingClassifier(
        estimators=modelos_base,
        final_estimator=meta_modelo,
        cv=cv,
        stack_method='predict_proba',
        n_jobs=-1,
        passthrough=passthrough,
    )

    # [FIX] StratifiedKFold explícito: garante representación de todas as
    # clases en cada fold. Especialmente importante para a clase 3 (5% dos
    # datos), que con un KFold aleatorio podería non aparecer nalgún fold.
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    # ── Validación cruzada do pipeline completo ──────────────────────────────
    print("Executando validación cruzada do stacking...")
    cv_scores = cross_val_score(
        stacking_clf, X_train, y_train,
        cv=skf,
        scoring='f1_macro',
        n_jobs=1,   # evitar conflito de paralelismo co n_jobs=-1 interno
    )
    print(f"  CV F1-Macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Por fold:   {np.round(cv_scores, 4)}\n")

    # ── Adestramento final sobre todos os datos de train ─────────────────────
    print("Adestramento final do stacking...")
    stacking_clf.fit(X_train, y_train)

    preds_train = stacking_clf.predict(X_train)
    f1_train = f1_score(y_train, preds_train, average='macro')
    print(f"  Train F1-Macro: {f1_train:.4f}")

    # Reporte por clase para detectar se aínda se ignoran clases minoritarias
    f1_por_clase = f1_score(y_train, preds_train, average=None)
    for i, f1 in enumerate(f1_por_clase):
        print(f"    · Clase {i}: F1 = {f1:.4f}")
    print()

    # ── Predicións sobre test ─────────────────────────────────────────────────
    preds_test = stacking_clf.predict(X_test)

    return stacking_clf, preds_test


# ─────────────────────────────────────────────────────────────────────────────
# Resto de funcións de modelado (sen cambios relevantes)
# ─────────────────────────────────────────────────────────────────────────────

def adestrar_por_bagging(
    modelo_base,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_estimators: int = 100,
    max_samples: float = 1.0,
    max_features: float = 1.0,
    cv: int = 5,
    seed: int = 42,
) -> tuple[object, np.ndarray]:
    """
    Adestra un clasificador mediante bagging sobre un modelo base.

    Parámetros
    ----------
    modelo_base   : Estimador sklearn que se usará como base do bagging.
    X_train       : Features de adestramento.
    y_train       : Target de adestramento.
    X_test        : Features de test.
    n_estimators  : Número de modelos base a adestrar.
    max_samples   : Fracción de mostras para cada modelo base.
    max_features  : Fracción de features para cada modelo base.
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
        bootstrap=True,
        bootstrap_features=False,
        n_jobs=-1,
        random_state=seed,
    )

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    print("Executando validación cruzada do bagging...")
    cv_scores = cross_val_score(bagging_clf, X_train, y_train, cv=skf, scoring='f1_macro')
    print(f"  CV F1-Macro (Bagging): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

    print("Adestramento final do bagging...")
    bagging_clf.fit(X_train, y_train)

    preds_train = bagging_clf.predict(X_train)
    f1_train = f1_score(y_train, preds_train, average='macro')
    print(f"  Train F1-Macro: {f1_train:.4f}\n")

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
    seed: int = 42,
) -> tuple[object, np.ndarray]:
    """
    Adestra un clasificador mediante boosting (gradient ou adaboost).

    Parámetros
    ----------
    X_train       : Features de adestramento.
    y_train       : Target de adestramento.
    X_test        : Features de test.
    tipo          : 'gradient' ou 'adaboost'.
    n_estimators  : Número de estimadores.
    learning_rate : Taxa de aprendizaxe.
    max_depth     : Profundidade máxima (só 'gradient').
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
            random_state=seed,
        )
    elif tipo == 'adaboost':
        boosting_clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=seed,
            algorithm='SAMME',
        )
    else:
        raise ValueError(f"Tipo non recoñecido: '{tipo}'. Usa 'gradient' ou 'adaboost'.")

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    print(f"Executando validación cruzada do boosting ({tipo})...")
    cv_scores = cross_val_score(boosting_clf, X_train, y_train, cv=skf, scoring='f1_macro')
    print(f"  CV F1-Macro (Boosting/{tipo}): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

    print(f"Adestramento final do boosting ({tipo})...")
    boosting_clf.fit(X_train, y_train)

    preds_train = boosting_clf.predict(X_train)
    f1_train = f1_score(y_train, preds_train, average='macro')
    print(f"  Train F1-Macro: {f1_train:.4f}\n")

    preds_test = boosting_clf.predict(X_test)
    return boosting_clf, preds_test


def adestrar_por_blending(
    modelos_base: list,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    meta_modelo=None,
    test_size: float = 0.2,
    passthrough: bool = True,
    seed: int = 42,
) -> tuple[object, np.ndarray]:
    """
    Adestra un conxunto de modelos base e un meta-modelo mediante blending,
    utilizando PROBABILIDADES (predict_proba) no canto de predicións duras.
    """
    if meta_modelo is None:
        meta_modelo = LogisticRegression(
            max_iter=1000,
            random_state=seed,
            class_weight='balanced',
        )

    # 1. Partición Holdout
    X_base, X_holdout, y_base, y_holdout = train_test_split(
        X_train, y_train,
        test_size=test_size,
        random_state=seed,
        stratify=y_train,
    )

    meta_features_holdout = []
    meta_features_test = []

    # 2. Adestramento dos modelos base e extracción de probabilidades
    for i, (nome, modelo) in enumerate(modelos_base):
        print(f"Adestrando co modelo {nome}")
        modelo.fit(X_base, y_base)
        
        # Usamos predict_proba. Isto devolve unha matriz de (N_mostras, 4 clases)
        probabilidades_holdout = modelo.predict_proba(X_holdout)
        probabilidades_test = modelo.predict_proba(X_test)
        
        meta_features_holdout.append(probabilidades_holdout)
        meta_features_test.append(probabilidades_test)

    # Xuntamos todas as matrices de probabilidades en horizontal
    meta_features_holdout = np.hstack(meta_features_holdout)
    meta_features_test = np.hstack(meta_features_test)

    # 3. Preparación do dataset final para o meta-modelo
    if passthrough:
        # Engadimos as features orixinais ás probabilidades
        X_holdout_arr = X_holdout.values if isinstance(X_holdout, pd.DataFrame) else X_holdout
        X_test_arr    = X_test.values    if isinstance(X_test,    pd.DataFrame) else X_test
        
        X_holdout_meta = np.hstack((X_holdout_arr, meta_features_holdout))
        X_test_meta    = np.hstack((X_test_arr,    meta_features_test))
    else:
        # O meta-modelo só ve as probabilidades
        X_holdout_meta = meta_features_holdout
        X_test_meta    = meta_features_test

    # 4. Adestramento do meta-modelo e avaliación
    print("Adestrando o meta-modelo")
    meta_modelo.fit(X_holdout_meta, y_holdout)

    preds_holdout_meta = meta_modelo.predict(X_holdout_meta)
    f1_holdout = f1_score(y_holdout, preds_holdout_meta, average='macro')
    print(f"F1-Macro do meta-modelo no Holdout: {f1_holdout:.4f}")

    # Reporte por clase no Holdout
    f1_por_clase = f1_score(y_holdout, preds_holdout_meta, average=None)
    for i, f1 in enumerate(f1_por_clase):
        print(f"\t· Clase {i}: F1 = {f1:.4f}")

    # 5. Predicións finais
    preds_test = meta_modelo.predict(X_test_meta)
    return meta_modelo, preds_test