from catboost import CatBoostClassifier

# Factoría para non repetir a configuración base de CatBoost en cada sitio
def crear_catboost(seed, depth, iterations, lr, l2, balance=None):
    # balance acepta 'Balanced' para compensar desbalance de clases, ou None para ignoralo
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=lr,
        depth=depth,
        l2_leaf_reg=l2,
        auto_class_weights=balance,
        random_seed=seed,
        verbose=0,
        early_stopping_rounds=40  # para automáticamente se non mellora en 40 iteracións
    )