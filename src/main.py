import pandas as pd
from preprocessing import prep, preprocessing

path = "Data/Telco_customer_churn.csv"
data = pd.read_csv((path))
processed = prep(data)
preprocessing(data).to_csv("Data/processed.csv")

