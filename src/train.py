from feature_engineering import pipe
from sklearn.model_selection import train_test_split
from main import processed
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from Improved_model import resample_train, improved_pipe
import numpy as np
import joblib
data = processed

# First model
pipeline, x, y = pipe(data)
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0, test_size=0.3)
pipeline.fit(x_train, y_train)




# Improved with Oversampling
imp_x_train, imp_y_train = resample_train(x_train, y_train)
improved_pipe, _, p , preprocessor= improved_pipe(data)
improved_pipe.fit(imp_x_train, imp_y_train)

if __name__ == '__main__':
    # Evaluation base model
    print("ROC_AUC:",roc_auc_score(y_test, pipeline.predict(x_test)))
    print("Accuracy:", pipeline.score(x_test, y_test))
    print(classification_report(y_test, pipeline.predict(x_test)))

    # Evaluation
    print('Improved')
    print("ROC_AUC:",roc_auc_score(y_test, improved_pipe.predict(x_test)))
    print("Accuracy:", improved_pipe.score(x_test, y_test))
    print(classification_report(y_test, improved_pipe.predict(x_test)))

    print('Threshold Tuned')
    y_proba = pipeline.predict_proba(x_test)[:,1]
    threshold = np.arange(0.1, 0.9, 0.01)

    best_f1 =0
    best_threshold = 0.5
    for t in threshold:
        y_pred = (y_proba >= t).astype('int')
        score = f1_score(y_test, y_pred)

        if score > best_f1:
            best_f1 = score
            best_threshold = t

    print('Best Threshold:', best_threshold)
    print('Best F1', best_f1)

    
    
    # Observing the derived threshold
    y_pred = (y_proba >= 0.54).astype('int')
    print(classification_report(y_test, y_pred))

    joblib.dump({'model': pipeline}, 'Model/churn_model.pkl')