# VA53-ACP
 Premier mini-projet de l'UV VA53, lié au cursus à l'UTBM, utilisant le machine learning et l'analyse par composantes principales pour la détection de visage.


# DataSet
Le dataset utilisé est l'ensemble des photos prises des étudiants. Pour faire fonctionner le programme, il faut placer l'ensemble des photos dans un dossier nommé "DataSetPostTraitement" lui même présent dans un dossier "DataSet" présent à la racine du projet.
Aprés avoir créé dans ce dossier deux sous dossiers appelés "train" et "test" il faut lancer le script "divisionDataSet.py" qui va diviser le dataset en deux ensembles de données, un pour l'entrainement et un pour le test.
Les photos doivent impérativement suivre la nomenclature suivante : "classe_nomDeLImage.jpg".

# Les scripts python
Les trois scripts python disponibles sont :
- "divisionDataSet.py" : permet de diviser le dataset en deux ensemble de données, un pour l'entrainement et un pour le test.
- "ACP.py" : permet de faire l'analyse par composantes principales sur le dataset.
- "machineLearning.py" : permet de faire l'entrainement du réseaux de neuronnes et de tester le modèle.