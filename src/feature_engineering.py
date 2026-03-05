import  pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose  import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from preprocessing import prep

def pipe(data):
    x = data.drop('Churn Value', axis = 1)
    y = data['Churn Value']

    numcols = x.select_dtypes(include=['int64', 'float64']).columns.tolist()
    catcols = x.select_dtypes(include = ['object']).columns.tolist()

    num_pipe = Pipeline([('scaler', StandardScaler())])
    cat_pipeline = Pipeline([('Encoder', OneHotEncoder(handle_unknown= 'ignore'))])

    preprocessor = ColumnTransformer([('num', num_pipe, numcols), ('cat', cat_pipeline, catcols)])

    model_pipeline = Pipeline([
        ('preprocessing', preprocessor), 
        ('classifier', LogisticRegression(class_weight='balanced', random_state=42))
        ])
    return model_pipeline, x,y