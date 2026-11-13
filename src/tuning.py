from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# from imblearn.over_sampling import SMOTE
# import xgboost as xgb
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
# from xgboost import XGBClassifier
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from src.transformers import create_features_preprocess
import pandas as pd
from src.visualisation import roc_auc_curve, cm_reference_models
from sklearn.model_selection import GridSearchCV
from src.evaluate import evaluate_models
import joblib

def tuning_models(X_train, X_test, y_train, y_test):

      results_Grid_Search = []

      models_with_params = {
            "LogisticRegression" : (
                  LogisticRegression(max_iter=3000, random_state=42),
                  {
                        'classificator__fit_intercept' : [True, False],
                        'classificator__l1_ratio' : [0.0],
                        'classificator__solver': ['saga'],
                        'classificator__max_iter': [10, 50],
                  },
                  True),
            "DecisionTreeClassifier" : (
            
                  DecisionTreeClassifier(random_state=42),
                  {
                        'classificator__max_depth': [3, 5, 7, 10, 15, 20, None],
                        'classificator__min_samples_split': [2, 5, 10, 20],
                        'classificator__min_samples_leaf': [1, 2, 5, 10],
                        'classificator__criterion': ["gini", "entropy"]
                  },
                  False
            ),
            
            "RandomForestClassifier": (
                  RandomForestClassifier( n_jobs=1, random_state=42),
                  {
                        'classificator__n_estimators': [100, 200, 300],
                        'classificator__max_depth': [5, 10, 15, 20, None],
                        'classificator__min_samples_split': [2, 5, 10],
                        'classificator__min_samples_leaf': [1, 2, 4, 8],
                        'classificator__max_features': ['sqrt', 'log2'],
                        'classificator__criterion': ['gini', 'entropy']
                  },
                  False
            ),
            "SVC" : (
                  SVC(kernel='linear', probability=True),
                  {
                        'classificator__C' : [0.1, 1],
                        'classificator__gamma' : ['scale', 'auto'],
                        'classificator__kernel' : ['linear', 'poly', 'rbf'],
                  },
                  True
            )
      }

      for model_name, (model, params, scaling) in models_with_params.items():

            features_preprocessor = create_features_preprocess(scaling=scaling)

            pipe = Pipeline([
            ('preproces', features_preprocessor),
            # ('smote', SMOTE(random_state=42)),
            ('classificator', model)
            ])


            gs = GridSearchCV(estimator=pipe, param_grid=params, cv=5, scoring='f1_macro', n_jobs=2, verbose=1,error_score="raise")

            # Fit
            gs.fit(X_train, y_train)

            # Best_Estimator
            best_estimator = gs.best_estimator_

            y_test_bin = label_binarize(y_test, classes=best_estimator.classes_)

            y_pred = best_estimator.predict(X_test)
            # Probabilités pour chaque classe
            y_score = best_estimator.predict_proba(X_test)
            
      
            #ROC_AUC
            roc_auc_curve(y_test=y_test, y_test_bin=y_test_bin, y_score=y_score, model_name=model_name, classes=best_estimator.classes_)

            # Matrice de confusion
            cm_reference_models(y_test=y_test, y_pred=y_pred, labels=best_estimator.classes_, model_name=model_name)

            # Evaluation du model
      
            # Rapport de classification
            print(classification_report(y_test, y_pred))

            # Résultats
            results_Grid_Search.append({
                  'model_name': model_name,
                  'model' : best_estimator,
                  'accuracy': accuracy_score(y_test, y_pred),
                  'recall_macro': recall_score(
                        y_test, y_pred, average='macro'
                  ),
                  'f1_macro': f1_score(
                        y_test, y_pred, average='macro'
                  )
            })

      # Comparaison finale
      results_Grid_Search_df = pd.DataFrame(results_Grid_Search).sort_values(
                  by='f1_macro',
                  ascending=False, ignore_index=True
            )

      best_model_CV = results_Grid_Search_df.iloc[0]['model']

      best_model_CV_name = results_Grid_Search_df.iloc[0]['model_name']

      y_pred = best_model_CV.predict(X_test)

      # Evaluation des données de Test
      
      evaluate_models(y_test=y_test, y_pred=y_pred, model_name=best_model_CV_name, best_model=best_model_CV)

      serialisaiton(best_model_CV)

      print("Fine_tuning met avec success! ")


def serialisaiton(best_model):
      joblib.dump(best_model, 'models/best_model.joblib')
      print("Serialisaiton met avec success!")