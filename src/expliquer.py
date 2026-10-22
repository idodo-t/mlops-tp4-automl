# Bonus - Exercice 1 : quelles variables comptent, pour quel modele ?
import pandas as pd
from flaml import AutoML
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

df = pd.read_csv('data/transactions.csv')
X, y = df.drop(columns='fraude'), df['fraude']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

automl = AutoML()
automl.fit(Xtr, ytr, task='classification', metric='f1', time_budget=60,
           seed=42, verbose=0)
modeles = {
    'automl-' + automl.best_estimator: automl.model.estimator,
    'manuel-rf-500-16': RandomForestClassifier(n_estimators=500, max_depth=16,
                                               random_state=42).fit(Xtr, ytr),
}
for nom, m in modeles.items():
    print(f'\n{nom} | F1 test : {f1_score(yte, m.predict(Xte)):.4f}')
    # on melange une colonne a la fois et on mesure la perte de F1
    r = permutation_importance(m, Xte, yte, scoring='f1', n_repeats=10,
                               random_state=42)
    for col, moy, ec in sorted(zip(X.columns, r.importances_mean,
                                   r.importances_std), key=lambda t: -t[1]):
        print(f'  {col:12s} perte de F1 = {moy:.3f} (+/- {ec:.3f})')