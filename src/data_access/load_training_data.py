import pandas as pd

def load_training_data(engine):
    query = "SELECT * FROM training_dataset"
    return pd.read_sql(query, engine)