import  pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose  import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample


def improved_pipe(data):
    x = data.drop('Churn Value', axis = 1)
    y = data['Churn Value']

    numcols = x.select_dtypes(include=['int64', 'float64']).columns.tolist()
    catcols = x.select_dtypes(include = ['object']).columns.tolist()

    num_pipe = Pipeline([('scaler', StandardScaler()), ('imputer', SimpleImputer(strategy = 'median'))])
    cat_pipeline = Pipeline([('Encoder', OneHotEncoder(handle_unknown= 'ignore')), 
                             ('imputer', SimpleImputer(strategy = 'most_frequent'))])

    preprocessor = ColumnTransformer([('num', num_pipe, numcols), ('cat', cat_pipeline, catcols)])

    model_pipeline = Pipeline([
        ('preprocessing', preprocessor), 
        ('classifier', LogisticRegression(class_weight= 'balanced', random_state = 42))
        ])
    return model_pipeline, x, y, preprocessor

def resample_train(x,y):
    df = pd.concat([x,y], axis = 1)
    df_major = df[df['Churn Value'] == 0]
    df_minor = df[df['Churn Value'] == 1]
    minor_ups = resample(df_minor, replace = True, n_samples= len(df_major), random_state=42)
    new_df = pd.concat([df_major, minor_ups])
    new_df = new_df.sample(frac = 1, random_state = 42)
    new_x = new_df.drop('Churn Value', axis =1)
    new_y = new_df['Churn Value']
    return new_x, new_y