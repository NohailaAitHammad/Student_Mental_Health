from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# from imblearn.pipeline import Pipeline
from src.config import (RANDOM_STATE, NOMINALS_COLUMNS, ORDINALS_COLUMNS, NUMERICS_COLUMNS, PHYSICAL_ACTIVITY_COLUMN, DIET_QUALITY_COLUMN)

class IQRClipper(BaseEstimator, TransformerMixin):
      def __init__(self, factor=1.5):
            self.factor = factor


      def fit(self, X, y=None):
            self.q1 = np.percentile(X, 25, axis=0)
            self.q3 = np.percentile(X, 75, axis=0)
            self.IQR = self.q3 - self.q1
            self.lower_bound_  = self.q1-(self.factor * self.IQR)
            self.upper_bound_ = self.q3 + (self.factor * self.IQR)
            return self

      def transform(self, X, y=None):
            X = np.asarray(X)
            X_clipped = np.clip(
                  X,
                  self.lower_bound_,
                  self.upper_bound_
            )

            return X_clipped


def create_features_preprocess(scaling=True):
      
      numerics_steps= [
                  ('imputer', SimpleImputer(strategy='median')),
                  ('iqr_clipper', IQRClipper()),
                  
            ]
      if scaling:
            numerics_steps.append(
                  ('scaler', StandardScaler(with_mean=False))
            )

      numerics_transformers = Pipeline(steps=numerics_steps)

      nominals_transformers = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('on_hot_encoder', OneHotEncoder()),
      ])

      Physical_Activity_transformers = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal_encoder', OrdinalEncoder( categories=[['Low', 'Moderate', 'High']])),
      ])

      Diet_Quality_transformers = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal_encoder', OrdinalEncoder(categories=[['Poor', 'Average', 'Good']])),
      ])

      features_preprocessor = ColumnTransformer([
            ('numerics', numerics_transformers, NUMERICS_COLUMNS),
            ('nominals', nominals_transformers, NOMINALS_COLUMNS),
            ('Physical_Activity', Physical_Activity_transformers, PHYSICAL_ACTIVITY_COLUMN),
            ('Diet_Quality', Diet_Quality_transformers, DIET_QUALITY_COLUMN)
            ],
            remainder='drop'
      )

      return features_preprocessor