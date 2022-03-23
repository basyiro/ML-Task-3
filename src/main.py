import sqlite3
import pandas as pd

# Local imports
import feature_engineering
import models
import preprocessing

if __name__ == "__main__":
    conn = sqlite3.connect(r"../data/survive.db")
    patients_df = pd.read_sql_query("SELECT * from survive", conn)
    patients_df = preprocessing.preprocessing(patients_df)
    patients_df = feature_engineering.feature_engineering(patients_df)
    models.run_models(patients_df)
