import os
import random
import matplotlib.pyplot as plt

#Liste des images
dir : list[str] = os.listdir('DataSet/DataSetPostTraitement/')

def divisionParClass(dir : list[str]) -> dict[int,list[str]]:
    """Création dun dictionnaire avec comme clé le numéro de la classe et comme valeur la liste des images de cette classe"""
    Dataset : dict[int,list[str]] = {}
    for photo in dir :
        clas = photo.split('_')[0]
        if len(clas)>3 :
            continue
        if clas in Dataset and Dataset[clas]!=None:
            Dataset[clas].append(photo)  
        else:
            Dataset[clas] =[photo]

    print(Dataset)
    return Dataset

def comptageClass(Dataset : dict[int,list[str]]) -> list[tuple[int,int]]:
    """Compte le nombre d'image par classe"""
    compteur = []
    for key in Dataset:
        compteur.append((key,len(Dataset[key])))
    return compteur



#Récupération des images par classe
Dataset = divisionParClass(dir)
#Comptage des images par classe
compteur = comptageClass(Dataset)

for classe, nb in compteur:
    nbTest = int(nb*0.25)#25% des images sont pour le test
    print(classe, nbTest)
    #Création des dossiers de test et train
    for i in range(nbTest):#On prend 25% des images pour le test
        index = random.randint(0,len(Dataset[classe])-1)
        os.rename('DataSet/DataSetPostTraitement/'+Dataset[classe][index],'DataSet/DataSetPostTraitement/test/'+Dataset[classe][index])
        Dataset[classe].pop(index)
    for i in range(len(Dataset[classe])):#Le reste des images sont pour l'entrainement
        os.rename('DataSet/DataSetPostTraitement/'+Dataset[classe][i],'DataSet/DataSetPostTraitement/train/'+Dataset[classe][i])
