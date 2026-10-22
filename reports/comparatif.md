# Comparatif des approches - projet fraude (TP04)

| Approche | F1 test | Temps humain | Temps machine | Cout estime (source, date) | Explicabilite | Export / reversibilite |
|---|---|---|---|---|---|---|
| Logistique (ref.) | 0.7407 | ~2 min | 0.2 s | 0 (poste local) | Maximale | Total |
| Manuel (TP2) | 0.8669 | ~30 min | 2.31 s | 0 (poste local) | Moyenne/Bonne | Total |
| FLAML 60 s | 0.8409 | ~5 min | 60 s (123 essais) | 0 (poste local) | Moyenne/Basse | Total |
| Vertex AI AutoML | non mesure | ~1 h (compte, facturation, quotas, 1er job) - estimation | non mesure | 21.252 $/node-hour d'entrainement tabulaire (Vertex AI pricing, consulte le 22/10/2026) | Bonne (Vertex Explainable AI, indisponible sur modele exporte) | Partielle : export vers Cloud Storage + Docker, CPU x86 avec AVX (doc Vertex AI, consultee le 22/10/2026) |
| SageMaker Autopilot | non mesure | ~1 h (compte, facturation, quotas, 1er job) - estimation | non mesure | Canvas : 1.9 $/h de session + entrainement 2.03-4.89 $/h (SageMaker Canvas pricing, consulte le 22/10/2026) | Bonne (3 notebooks generes : exploration, candidats, model insights) | A verifier : notebooks reproductibles, pas de page d'export trouvee |
| Azure Automated ML | non mesure | ~1 h (compte, facturation, quotas, 1er job) - estimation | non mesure | Pas de frais pour le service, seulement calcul et stockage (Azure ML pricing, consulte le 22/10/2026) | Bonne (importances locales et globales via le SDK d'interpretabilite) | A verifier (doc Azure ML, a completer) |

## Commentaire (5 lignes)

1. Logistique : a garder comme reference et quand l'explicabilite prime (secteur regule, decision a justifier client par client).
2. Reglage manuel : justifie quand on connait bien les donnees et qu'on veut le meilleur score ; ici il reste le meilleur (0.8669 contre 0.8561 pour FLAML a 120 s).
3. FLAML local : justifie pour obtenir vite un bon modele (0.84 en 10 s) sur un poste local, sans cout ni donnees envoyees hors de l'entreprise.
4. Plateformes managees : justifiees quand les donnees sont deja dans le cloud, ou qu'il faut deployer, versionner et superviser a l'echelle ; attention au cout et au verrouillage.
5. Dans tous les cas, l'AutoML ne remplace pas le jugement : la fuite du bloc D (F1 1.0 puis 0.0) n'a ete vue par aucun outil.
