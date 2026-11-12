from src.load_data import data_loading
from src.preprocessing import preprocessing_data
from src.train import  training_models
from src.tuning import  tuning_models
from src.config import (RANDOM_STATE, DIET_QUALITY_COLUMN, NOMINALS_COLUMNS, NUMERICS_COLUMNS, ORDINALS_COLUMNS, PHYSICAL_ACTIVITY_COLUMN)

def main():
      # Cahrgement de data
      df = data_loading('data/raw/dataset.csv')
      print("Chargement success!")
      #Pretraitement des données
      X_train, X_test, y_train, y_test = preprocessing_data(df)
      print("division success!")
      #Entrainement des models de reference
      training_models(X_train, X_test, y_train, y_test)
      # Fine-tuning des hyperparametres
      tuning_models(X_train, X_test, y_train, y_test)

main()




