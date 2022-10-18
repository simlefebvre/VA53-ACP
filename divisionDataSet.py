import os
import random
import matplotlib.pyplot as plt


dir : list[str] = os.listdir('DataSet/DataSetPostTraitement/')

"""for photo in dir:
    if photo[:4] == 'test' and photo != 'test':
        os.rename('DataSet/DataSetPostTraitement/'+photo,'DataSet/DataSetPostTraitement/'+ photo[4:])"""

def divisionParClass(dir : list[str]) -> dict[int,list[str]]:
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
    compteur = []
    for key in Dataset:
        compteur.append((key,len(Dataset[key])))
    return compteur

Dataset = divisionParClass(dir)
compteur = comptageClass(Dataset)


for classe, nb in compteur:
    nbTest = int(nb*0.25)
    print(classe, nbTest)
    for i in range(nbTest):
        index = random.randint(0,len(Dataset[classe])-1)
        os.rename('DataSet/DataSetPostTraitement/'+Dataset[classe][index],'DataSet/DataSetPostTraitement/test/'+Dataset[classe][index])
        Dataset[classe].pop(index)
