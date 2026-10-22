import pandas as pd
from flaml import AutoML
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/transactions.csv')
X, y = df.drop(columns='fraude'), df['fraude']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

a = AutoML()
a.fit(Xtr, ytr, task='classification', metric='f1', time_budget=30,
      seed=42, verbose=0, estimator_list=['lrl1'])
pred = a.predict(Xte)
print('fraudes predites :', int(pred.sum()), '| fraudes reelles :', int(yte.sum()))
print('F1 validation :', round(1 - a.best_loss, 4))