from catboost import CatBoostClassifier

def crear_catboost(seed, depth, iterations, lr, l2, balance=None):
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=lr,
        depth=depth,
        l2_leaf_reg=l2,
        auto_class_weights=balance,
        random_seed=seed,
        verbose=0,
        early_stopping_rounds=40
    )