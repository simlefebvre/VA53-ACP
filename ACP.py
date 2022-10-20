import cv2 as cv
import numpy as np
import sys
import os
import time

#Lister les images du dossier images
Lnom = os.listdir("DataSet/DataSetPostTraitement/train/")

# Read image from file
def lecture_image(nom_image : str) -> np.ndarray:
    """A partir d'un nom de fichier renvoie l'image sous forme de tableau numpy"""
    image = cv.imread(nom_image,0)
    image = cv.resize(image, (2000, 2000), interpolation=cv.INTER_AREA)
    return image

def computeModel(nbComposante : int):
    """
    Calcul le modèle de reconnaissance et renvoie le dictionnaire des coordonées des image dans le repère des vecteurs propres, le vecteur de moyenne, la matrice des vecteurs propres 
    """
    Limg = [lecture_image(f"DataSet/DataSetPostTraitement/train/{nom_image}") for nom_image in Lnom] #Récupération des images
    LimgVect = [img.flatten() for img in Limg] #Vectorisation des images
    M = len(Limg)
    sumVect = np.add.reduce(LimgVect) #Somme des vecteurs
    meanVect = sumVect/M #Moyenne des vecteurs
    VectEcart = {nom_image : np.subtract(LimgVect[i],meanVect) for i,nom_image in enumerate(Lnom)} #Vecteur d'écart
    A = np.array([VectEcart[nom_image] for nom_image in Lnom]).T
    matAAT = np.dot(A.T,A) #Matrice AAt
    Ai, Vi = np.linalg.eig(matAAT) #Calcul des valeurs propres et vecteurs propres

    eig = list(zip(Ai,Vi))
    eig.sort(key=lambda x: x[0], reverse=True) #Tri des valeurs propres

    comp = eig[:nbComposante] #On ne garde que les nbComposante premières valeurs propres
    Ai = [comp[i][0] for i in range(nbComposante)]
    Vi = [comp[i][1] for i in range(nbComposante)]
    
    LVectPropre = [np.dot(A,vi) for vi in Vi] #Vecteurs propres
    dictValVectPropre = {Ai[i] : LVectPropre[i] for i in range(len(Ai))} #Dictionnaire des valeurs propres et des vecteurs propres
    matUi = np.array([dictValVectPropre[Ai[i]] for i in range(len(Ai))]) #Matrice des vecteurs propres
    dictohmegai = {nom : np.dot(matUi,VectEcart[nom]) for nom in VectEcart.keys()} #Vecteur des coordonnées des images dans le repère des vecteurs propres
    return dictohmegai,meanVect,matUi

def guess(image,dictohmegai,meanVect,matUi):
    """
    Réalise la prédiction de la classe de l'image image en fonction du modèle dictohmegai,meanVect,matUi
    """
    ecartMoyenne = np.subtract(image.flatten(),meanVect) #Vecteur d'écart entre l'image à reconnaitre et la moyenne
    vecteurPoids = np.dot(matUi,ecartMoyenne) #Vecteur des coordonnées de l'image à reconnaitre dans le repère des vecteurs propres

    #Comparaison des vecteurs poids
    imgPlusProche = None
    distPlusfaible = sys.maxsize
    imgPlusProche2 = None
    distPlusfaible2 = sys.maxsize
    imgPlusProche3 = None
    distPlusfaible3 = sys.maxsize


    for name, img in dictohmegai.items() : #Parcours des vecteurs poids des images
        dist = np.linalg.norm(img-vecteurPoids) #Calcul de la distance entre les deux vecteurs poids
        #Comparaison des distances et mise à jour des images les plus proches
        if dist < distPlusfaible : 
            distPlusfaible3 = distPlusfaible2
            imgPlusProche3 = imgPlusProche2
            distPlusfaible2 = distPlusfaible
            imgPlusProche2 = imgPlusProche
            distPlusfaible = dist 
            imgPlusProche = name
        elif dist < distPlusfaible2 : 
            distPlusfaible3 = distPlusfaible2
            imgPlusProche3 = imgPlusProche2
            distPlusfaible2 = dist 
            imgPlusProche2 = name
        elif dist < distPlusfaible3 : 
            distPlusfaible3 = dist 
            imgPlusProche3 = name
    return [imgPlusProche,imgPlusProche2,imgPlusProche3],[distPlusfaible,distPlusfaible2,distPlusfaible3]

def test(nbComposante : int) -> tuple[float, float, float]:
    """
    Test le modèle de reconnaissance sur les images du dossier test et retourne le temps d'exécution du calcul du modèle, le temps d'exécution de la prédiction et le taux de reconnaissance
    """
    start = time.time()
    dictohmegai,meanVect,matUi = computeModel(nbComposante) #Calcul du modèle
    dureeModel = time.time() - start
    start = time.time()
    Lnom = os.listdir("DataSet/DataSetPostTraitement/test/")
    compteur = 0
    for nom_image in Lnom:
        img,dist = guess(lecture_image("DataSet/DataSetPostTraitement/test/" + nom_image),dictohmegai,meanVect,matUi) #Prédiction
        if img[0].split('_')[0] == nom_image.split('_')[0] : #Comparaison avec la classe réelle
            compteur += 1
        print(f"L'image {nom_image} après être passée par l'ACP est plus proche des images {img} à une distance de respectivemnet {dist}")
    dureeImage = time.time()-start
    ratio = compteur/len(Lnom)
    return dureeImage,ratio,dureeModel


for i in range(1,68):
    print("Nombre de composantes : ",i)
    duree,ratio,dureeModel = test(i)
    print("Temps de calcul du modèle : ",dureeModel)
    print("Temps de calcul des images : ",duree)
    print("Pourcentage de réussite : ",ratio*100,"%")
    print("-----------------------------")