from sklearn.model_selection import train_test_split
from src.config import (RANDOM_STATE)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# import xgboost as xgb
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
def spliting_data(df_clean, RANDOM_STATE):

      # Division des variables et le target

      X = df_clean.drop(['Mental_Health_Risk'], axis=1)
      y = df_clean['Mental_Health_Risk']

      # Division des données en 80% / 20 %

      return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def preprocessing_data(df):

      # Copy de dataset original

      df_clean = df.copy()

      # Suppression des colonnes sans role

      df_clean = df_clean.drop(['Student_ID'], axis=1)

      # Suppression de l'incohérences métier entre les variables Social Media Hours vs Screen Time Hours

      df_clean = df_clean[
            df_clean['Social_Media_Hours'] < df_clean['Screen_Time_Hours']
      ]

      # Division de dataset et train / test

      X_train, X_test, y_train, y_test = spliting_data(df_clean=df_clean, RANDOM_STATE=RANDOM_STATE)

      return X_train, X_test, y_train, y_test





