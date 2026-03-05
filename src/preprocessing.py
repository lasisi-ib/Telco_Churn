import pandas as pd

def preprocessing(data):
    data['Total Charges'] = pd.to_numeric(data['Total Charges'], errors = 'coerce')
    data['Total Charges'].fillna(data['Total Charges'].mean(), inplace = True)
    for i in data.select_dtypes(include= 'object').columns:
        if len(data[i].value_counts()) <= 5:
            data[i]= encoder(i, data)
        else:
            pass
    return data

def encoder(var, data):
    temp = pd.DataFrame(data[var].value_counts()).reset_index()
    data[var]= data[var].apply(lambda a: temp[temp[var]== a].index[0])
    return data[var]

def prep(data):
    data['Total Charges'] = pd.to_numeric(data['Total Charges'], errors = 'coerce')
    data['Total Charges'].fillna(data['Total Charges'].mean(), inplace = True)
    return data[['Senior Citizen','Contract','Payment Method', 'Multiple Lines',
                    'Phone Service','Tenure Months', 'Monthly Charges', 'Total Charges',
                    'CLTV', 'Churn Value']]
