# from datetime import datetime, timedelta
# from airflow.decorators import dag, task


# @dag(
#       dag_id="StudentMentelHealth",
#       #start_date=datetime(2026,11,12),
#       # shedule='@daily',
# )

# def StudentMentelHealth():

#       @task()

#       def training():
#             print("heelo")

#       training()


# StudentMentelHealth()


from datetime import datetime
from airflow.decorators import dag, task


@dag(
    dag_id="StudentMentelHealth",
    start_date=datetime(2026, 11, 12),
    schedule=None,  # Exécution manuelle uniquement
    catchup=False,
    tags=["test"],
)
def StudentMentelHealth():

    @task()
    def training():
        print("Hello from Airflow!")
        print("The DAG is running successfully.")
        print("Test completed!")

    training()


StudentMentelHealth()

