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

from src.evaluate import evaluate_models

def training_models(X_train, X_test, y_train, y_test):

      results = []
      models = {
            "LogisticRegression" : (LogisticRegression(max_iter=1000, random_state=42), None, True),
            "DecisionTreeClassifier" : (DecisionTreeClassifier(max_depth=10, random_state=42), None, False),
            "RandomForestClassifier" : (RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42), None, False),
            "SVC" : (SVC(kernel='rbf', probability=True), None, True)
      }

      for model_name, (model, params, scaling) in models.items():

            features_preprocessor = create_features_preprocess(scaling=scaling)
            
            pipe = Pipeline([
                  ('preprocess', features_preprocessor),

                  # ('smote', SMOTE(random_state=42)),

                  ('classificator', model)
            ])

            # Fit 
            pipe.fit(X_train, y_train)

            # Label Binary for the classes

            y_test_bin = label_binarize(y_test, classes=pipe.classes_)

            # Prediction des données de test

            y_pred = pipe.predict(X_test)

            # Prediction des Probabilités pour chaque classe des données de test

            y_score = pipe.predict_proba(X_test)

            #ROC_AUC
            roc_auc_curve(y_test=y_test, y_test_bin=y_test_bin, y_score=y_score, model_name=model_name, classes=pipe.classes_)

            # Matrice de confusion
            cm_reference_models(y_test=y_test, y_pred=y_pred, labels=pipe.classes_, model_name=model_name)

            # Evaluation du model

            # Rapport de classification
            print(classification_report(y_test, y_pred))
            
            # Resultats
            results.append({
                        'model_name': model_name,
                        'model': pipe,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'recall_macro': recall_score(
                              y_test, y_pred, average='macro'
                        ),
                        'f1_macro': f1_score(
                              y_test, y_pred, average='macro'
                        )
                  })

      # Comparaison finale
      results_df = pd.DataFrame(results).sort_values(
            by='f1_macro',
            ascending=False, ignore_index=True
      )

      best_model_reference = results_df.iloc[0]['model']
      best_model_reference_name = results_df.iloc[0]['model_name']

      y_pred = best_model_reference.predict(X_test)

      # Evaluation des données de Test

      evaluate_models(y_test=y_test, y_pred=y_pred, model_name=best_model_reference_name, best_model=best_model_reference)

      print("Entrainement reference met avec success! ")
