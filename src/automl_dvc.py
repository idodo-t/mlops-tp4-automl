# Bonus - Exercice 2 : la recherche AutoML comme etape d'un pipeline DVC
import json
from pathlib import Path
import joblib
import yaml
import pandas as pd
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

p = yaml.safe_load(open('params.yaml', encoding='utf-8'))['automl']
df = pd.read_csv('data/transactions.csv')
X, y = df.drop(columns='fraude'), df['fraude']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

a = AutoML()
a.fit(Xtr, ytr, task='classification', metric='f1',
      time_budget=p['budget'], seed=p['seed'], verbose=0)
f1 = round(f1_score(yte, a.predict(Xte)), 4)

Path('models').mkdir(exist_ok=True)
Path('reports').mkdir(exist_ok=True)
joblib.dump(a.model.estimator, 'models/automl.pkl')
json.dump({'f1': f1, 'famille': a.best_estimator},
          open('reports/metrics_automl.json', 'w'), indent=2)
print('automl :', a.best_estimator, '| F1 test =', f1)