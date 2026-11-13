from datetime import datetime, timedelta
from airflow.decorators import dag, task
from pathlib import Path
import pickle

@dag(
      dag_id="Student_Mentel_Health",
      start_date=datetime(2026, 11, 12),
      schedule=None,  # Exécution manuelle uniquement
      default_args={
            "retries": 3,
            "retry_delay": timedelta(minutes=5),
    },
)
def Student_Mentel_Health():

      @task()
      def loading_dataset():
            from src.load_data import data_loading
            print("Loading Dataset!")
            raw_path = "/opt/airflow/data/raw/dataset.csv"
            output_path = Path("/opt/airflow/data/processed/loaded_data.pkl")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df = data_loading(raw_path)

            with open(output_path, "wb") as file:
                  pickle.dump(df, file)

            print("Loading completed!")

            return str(output_path)
      
      @task
      def preprocessing_dataset(loaded_path):
            from src.preprocessing import preprocessing_data
            print("Preprocessing Dataset!")
            with open(loaded_path, "rb") as file:
                  df = pickle.load(file)

            X_train, X_test, y_train, y_test = preprocessing_data(df)

            output_path = "/opt/airflow/data/processed/preprocessed_data.pkl"

            with open(output_path, "wb") as file:
                  pickle.dump(
                  (X_train, X_test, y_train, y_test),
                  file
                  )

            print("Preprocessing completed!")
            return str(output_path)
      
      @task
      def training_models_reference(processed_path):
            from src.train import  training_models
            print("Training Models Reference!")
            with open(processed_path, "rb") as file:
                  X_train, X_test, y_train, y_test = pickle.load(file)

            training_models(X_train, X_test, y_train, y_test)

            print("Training Reference Models completed!")

            return str(processed_path)
      @task
      def fine_tuning_models(processed_path):
            from src.tuning import  tuning_models
            print("Fine_Tuning Models !")
            with open(processed_path, "rb") as file:
                  X_train, X_test, y_train, y_test = pickle.load(file)

            tuning_models(X_train, X_test, y_train, y_test)

            print("Fine-Tuning Models completed!")


      extracted = loading_dataset()

      processed = preprocessing_dataset(extracted)

      trained = training_models_reference(processed)

      fine_tuning_models(trained)


Student_Mentel_Health()
