from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from config import CV_FOLDS, RANDOM_STATE


def build_model_pipelines(preprocessor):
    baseline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    ])

    rf = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE))
    ])

    gb = Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))
    ])

    return baseline, rf, gb


def tune_random_forest(rf_pipeline, X_train, y_train):
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5]
    }

    search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=CV_FOLDS,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search


def tune_gradient_boosting(gb_pipeline, X_train, y_train):
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3]
    }

    search = GridSearchCV(
        estimator=gb_pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=CV_FOLDS,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search
