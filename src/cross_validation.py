from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from train import x, y , preprocessor
from sklearn.pipeline import Pipeline

models = {'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42),
           "GradientBoosting":GradientBoostingClassifier(random_state=42), 
          "LogisticRegression": LogisticRegression(class_weight='balanced', random_state=42), 
          "SupportVectorClassifier":SVC(probability=True, class_weight='balanced', random_state=42)}
metrics = {"Accuracy": 'accuracy', 'Precision':'precision', 'F1': 'f1', 'Recall':'recall' ,'Roc_Auc':'roc_auc'}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state= 42)

for name, model in models.items():
    pipeline_cv = Pipeline([('preprocessor',preprocessor), ('Classifier', model)])
    result = cross_validate(pipeline_cv, x, y, cv=cv, scoring = metrics,  )
    print(f"\n{name}")
    for metric in metrics:
        print(metric, ":", result['test_'+metric].mean())