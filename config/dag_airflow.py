from datetime import datetime, timedelta
from airflow.decorators import dag, task
from src.load_data import data_loading

@dag(
      dag_id="StudentMentelHealth",
      #start_date=datetime(2026,11,12),
      shedule='@daily',
)

def StudentMentelHealth():

      @task(retrie=3,
            retry_delay=timedelta(minutes=1)
      )

      def training():
            return data_loading()

      training = training()
      training

StudentMentelHealth()