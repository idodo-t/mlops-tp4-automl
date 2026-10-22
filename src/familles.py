# Bonus - Exercice 1 (suite) : le prix de la simplicite
import sys
import pandas as pd
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

df = pd.read_csv('data/transactions.csv')
X, y = df.drop(columns='fraude'), df['fraude']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

for familles in (['lrl1'], ['rf', 'extra_tree'], 'auto'):
    a = AutoML()
    a.fit(Xtr, ytr, task='classification', metric='f1', time_budget=30,
          seed=42, verbose=0, estimator_list=familles)
    f1 = f1_score(yte, a.predict(Xte))
    print(f'{str(familles):25s} -> {a.best_estimator:12s} F1 test = {f1:.4f}')