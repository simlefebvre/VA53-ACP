import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from nameList import class_names

print("Fin des importations")

# Read image from file
def lecture_image(nom_image : str,input : int = 250) -> np.ndarray:
    """A partir d'un nom de fichier renvoie l'image sous forme de tableau numpy"""
    image = cv.imread(nom_image,0)
    img = cv.resize(image, (input, input), interpolation=cv.INTER_AREA)
    return img

def extractImagesAndLabels(path : str,input : int = 250,externe=False) -> list:
    """Récupére les informations du DataSet"""
    LNomTrain = os.listdir(f"{path}/train/")

    train_images = np.array([lecture_image(f"{path}/train/{nom_image}",input) for nom_image in LNomTrain]) #Récupération des images
    train_labels = np.array([int(nom_image.split("_")[0]) for nom_image in LNomTrain]) #Récupération des labels

    if externe: #Si on veut tester sur des images externes
      Lnom = os.listdir(f"{path}/test_externe")
      test_images = np.array([lecture_image(f"{path}/test_externe/{nom_image}",input) for nom_image in Lnom])
      test_labels = np.array([int(nom_image.split("_")[0]) for nom_image in Lnom])
    else: #Sinon on teste sur les images de test
      Lnom = os.listdir(path+"/test/" )
      test_images = np.array([lecture_image(path+"/test/"+ nom_image,input) for nom_image in Lnom]) #Récupération des images
      test_labels = np.array([int(nom_image.split("_")[0]) for nom_image in Lnom]) #Récupération des labels

    #Normalisation des images
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return train_images, train_labels, test_images, test_labels

def generateModel(train_images, train_labels, nbEpochs, nbDense = 1, nbNeurone = [512],numeroModel=1,input=250,callbacksEnable=False) -> tf.keras.Sequential:
    """Création du modèle avec les paramètres donnés"""

    #Construire le modèle
    #Ajout de la couche d'entrée
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(input, input)),
    ])
    #Ajout des couches cachées
    for i in range(nbDense):
        model.add(tf.keras.layers.Dense(nbNeurone[i], activation='relu'))

    #Ajout de la couche de sortie
    model.add(tf.keras.layers.Dense(15)) 

    #Compilation du modèle
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    
    callbacks = [tf.keras.callbacks.TensorBoard(
    log_dir='my_log_dir/{}'.format(numeroModel),
    histogram_freq=1,
    embeddings_freq=1,
    )]

    #Choix de sauvegarder ou non les résultats
    if callbacksEnable:
      #Entrainer le modèle
      model.fit(train_images, train_labels, epochs=nbEpochs, callbacks=callbacks)
    else:
      #Entrainer le modèle
      model.fit(train_images, train_labels, epochs=nbEpochs)

    return model

def makePrediction(model, test_images) -> tf.keras.Sequential:
    """Calculer les prédictions"""
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    return predictions

def plot_image(i, predictions_array, true_label, img):
  """Affiche l'image et la prédiction"""
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap='gray')

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  """Affiche le tableau des prédictions"""
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(15))
  plt.yticks([])
  thisplot = plt.bar(range(15), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')





#Variables 
nbEpochs = 45
tailleImage = 50
nbNeurone = 1000
nbCouche = 2

#Récupération des données
train_images, train_labels, test_images, test_labels = extractImagesAndLabels("DataSet/DataSetPostTraitement",tailleImage,externe=False)

#Création et entrainement du modèle
iteration = 0
model = generateModel(train_images, train_labels, nbEpochs,nbDense=nbCouche,nbNeurone=[nbNeurone]*nbCouche,input=tailleImage,callbacksEnable=False)

#Calcul des prédictions
predictions = makePrediction(model, test_images)

#Affichage des prédictions
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(3*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()