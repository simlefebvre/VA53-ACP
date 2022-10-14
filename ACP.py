import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os
import time

#Lister les images du dossier images
Lnom = os.listdir("DataSet/DataSetPostTraitement/")
imageCherchee = "test/0_IMG_2165_R.jpeg"
Lnom.remove('test')


# Read image from file
def lecture_image(nom_image : str) -> np.ndarray:
    """A partir d'un nom de fichier renvoie l'image sous forme de tableau numpy"""
    image = cv.imread(f"DataSet/DataSetPostTraitement/{nom_image}",0)
    return image

def computeModel():
    """
    Calcul le modèle de reconnaissance et renvoie le dictionnaire des coordonée des image dans le repère des vecteurs propres, le vecteur de moyenne, la matrice des vecteurs propres 
    """
    Limg = [lecture_image(nom_image) for nom_image in Lnom] #Récupération des images
    LimgVect = [img.flatten() for img in Limg] #Véctorisation des images
    M = len(Limg)
    sumVect = np.add.reduce(LimgVect) #Somme des vecteurs
    meanVect = sumVect/M #Moyenne des vecteurs
    VectEcart = {nom_image : np.subtract(LimgVect[i],meanVect) for i,nom_image in enumerate(Lnom)} #Vecteur d'écart
    A = np.array([VectEcart[nom_image] for nom_image in Lnom]).T
    matAAT = np.dot(A.T,A) #Matrice AAt
    Ai, Vi = np.linalg.eig(matAAT) #Calcul des valeurs propres et vecteurs propres
    LVectPropre = [np.dot(A,vi) for vi in Vi] #Vecteurs propres
    dictValVectPropre = {Ai[i] : LVectPropre[i] for i in range(len(Ai))} #Dictionnaire des valeurs propres et des vecteurs propres
    matUi = np.array([dictValVectPropre[Ai[i]] for i in range(len(Ai))]) #Matrice des vecteurs propres
    dictohmegai = {nom : np.dot(matUi,VectEcart[nom]) for nom in VectEcart.keys()} #Vecteur des coordonnées des images dans le repère des vecteurs propres
    return dictohmegai,meanVect,matUi

def guess(image,dictohmegai,meanVect,matUi):
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

def test():
    start = time.time()
    dictohmegai,meanVect,matUi = computeModel()
    print("Temps de calcul du modèle : ",time.time()-start)
    start = time.time()
    Lnom = os.listdir("DataSet/DataSetPostTraitement/test")
    for nom_image in Lnom:
        img,dist = guess(lecture_image("/test/" + nom_image),dictohmegai,meanVect,matUi)
        print(nom_image,img,dist)
    print("Temps de calcul des images : ",time.time()-start)


test()