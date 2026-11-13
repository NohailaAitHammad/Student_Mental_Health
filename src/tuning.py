from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
      accuracy_score,
      recall_score,
      f1_score,
      classification_report
)

from sklearn.preprocessing import label_binarize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from src.transformers import create_features_preprocess
from src.visualisation import roc_auc_curve, cm_reference_models
from src.evaluate import evaluate_models

import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def tuning_models(X_train, X_test, y_train, y_test):

      results_tuning = []

      models_with_params = {

            "LogisticRegression": (
                  LogisticRegression(max_iter=3000, random_state=42),
                  {
                  'classificator__C': [0.01, 0.1, 1, 10],
                  'classificator__fit_intercept': [True, False],
                  'classificator__solver': ['lbfgs', 'saga'],
                  'classificator__class_weight': [None, 'balanced']
                  },
                  True
            ),

            "DecisionTreeClassifier": (
                  DecisionTreeClassifier(random_state=42),
                  {
                  'classificator__max_depth': [3, 5, 10, 15, 20, None],
                  'classificator__min_samples_split': [2, 5, 10, 20],
                  'classificator__min_samples_leaf': [1, 2, 5, 10],
                  'classificator__criterion': ['gini', 'entropy'],
                  'classificator__class_weight': [None, 'balanced']
                  },
                  False
            ),

            "RandomForestClassifier": (
                  RandomForestClassifier(random_state=42, n_jobs=-1),
                  {
                  'classificator__n_estimators': [100, 150, 200, 300],
                  'classificator__max_depth': [5, 10, 15, 20, None],
                  'classificator__min_samples_split': [2, 5, 10],
                  'classificator__min_samples_leaf': [1, 2, 4, 8],
                  'classificator__max_features': ['sqrt', 'log2'],
                  'classificator__criterion': ['gini', 'entropy'],
                  'classificator__class_weight': [None, 'balanced']
                  },
                  False
            ),

            "SVC": (
                  SVC(probability=True, random_state=42),
                  {
                  'classificator__C': [0.1, 1, 10],
                  'classificator__gamma': ['scale', 'auto'],
                  'classificator__kernel': ['linear', 'rbf'],
                  'classificator__class_weight': [None, 'balanced']
                  },
                  True
            )
      }

      for model_name, (model, params, scaling) in models_with_params.items():

            print(f"\n{'=' * 20} {model_name} {'=' * 20}")

            features_preprocessor = create_features_preprocess(
                  scaling=scaling
            )

            pipe = Pipeline([
                  ('preproces', features_preprocessor),
                  ('classificator', model)
            ])

            # Choisir la méthode de recherche
            if model_name == "LogisticRegression":

                  search = GridSearchCV(
                  estimator=pipe,
                  param_grid=params,
                  cv=3,
                  scoring='f1_macro',
                  n_jobs=-1,
                  verbose=1,
                  error_score='raise'
                  )

            else:

                  search = RandomizedSearchCV(
                  estimator=pipe,
                  param_distributions=params,
                  n_iter=10,
                  cv=3,
                  scoring='f1_macro',
                  random_state=42,
                  n_jobs=-1,
                  verbose=1,
                  error_score='raise'
                  )

            # Entraînement
            search.fit(X_train, y_train)

            best_estimator = search.best_estimator_

            print("Meilleurs paramètres :", search.best_params_)
            print("Meilleur F1 macro CV :", search.best_score_)

            # Prédictions
            y_pred = best_estimator.predict(X_test)
            y_score = best_estimator.predict_proba(X_test)

            y_test_bin = label_binarize(
                  y_test,
                  classes=best_estimator.classes_
            )

            # ROC-AUC
            roc_auc_curve(
                  y_test=y_test,
                  y_test_bin=y_test_bin,
                  y_score=y_score,
                  model_name=model_name,
                  classes=best_estimator.classes_
            )

            # Matrice de confusion
            cm_reference_models(
                  y_test=y_test,
                  y_pred=y_pred,
                  labels=best_estimator.classes_,
                  model_name=model_name
            )

            print(classification_report(y_test, y_pred))

            results_tuning.append({
                  'model_name': model_name,
                  'model': best_estimator,
                  'best_params': search.best_params_,
                  'cv_f1_macro': search.best_score_,
                  'accuracy': accuracy_score(y_test, y_pred),
                  'recall_macro': recall_score(
                  y_test, y_pred, average='macro'
                  ),
                  'f1_macro': f1_score(
                  y_test, y_pred, average='macro'
                  )
            })

      # Comparaison des modèles selon le F1 macro CV
      results_df = pd.DataFrame(results_tuning).sort_values(
            by='cv_f1_macro',
            ascending=False,
            ignore_index=True
      )

      print("\n===== Comparaison finale =====")
      print(results_df[
            ['model_name', 'cv_f1_macro', 'accuracy',
            'recall_macro', 'f1_macro']
      ])

      # Meilleur modèle
      best_model = results_df.iloc[0]['model']
      best_model_name = results_df.iloc[0]['model_name']

      y_pred = best_model.predict(X_test)

      evaluate_models(
            y_test=y_test,
            y_pred=y_pred,
            model_name=best_model_name,
            best_model=best_model
      )

      serialisaiton(best_model)

      print("\nFine-tuning terminé avec succès !")

      return results_df


def serialisaiton(best_model):
      joblib.dump(best_model, 'models/best_model.joblib')
      print("Sérialisation effectuée avec succès !")