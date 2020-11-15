# RedNeuronal_TFG
#### Inicio del proyecto día 08/10/2020

Proyecto de trabajo de fin de Grado de Daniel Fernández Alonso.
## Reconocimiento  de imágenes obtenidas con dron mediante técnicas de DeepLearning
Este proyecto va ser una reconocimeinto de imágenes procesadas con Deep Learning, para sacar patrones, obteniendo así posibles registros para identificar, por medio de imagenes, rasgos de caminos, lugares arqueologicos, y cruces de caminos.

# Librerías útiles 📖

## **Numpy**
* Soporte para vectores y matrices, funciones matemáticas para operaciones de alto nivel de vectores y matrices
* **Importación :** import numpy as np

## Scipy
* Herramientas y algoritmos matemáticos para optimización, álgebra lineal, etc, **procesamiento de señales e imagenes**  y tareas de ciencia e ingeniería
* **Importación:** _Añadir la base de importación_

## Pandas
* Análisis de datos, para limpiar los datos en bruto, y que sean aptos para el análisis. Tareas como alinear datos para compararlos, fusion de conjuntos de datos, gestion de datos perdidos. Es una librería para procesamiento de datos estadísticos
* **Importación:** from pandas import DataFrame

## Matplotlib
* Generación de gráficos a partir de listas o array de python y de numpy.
* **Importación:** import matplotlib.pyplot as plt

## Scikit-Learn
* Construida sobre Numpy, Scipy y Matplotlib, implementa muchos de los algoritmos de Marchine Learning
* **Importación:** from sklearn import **_Añadir la libreria que nos haga falta_**

## Keras
* Librería fácil para el usuario para usar el deep Learning
* **Importación:** from keras.**_libreria a añadir_** import **_lib a añadir_**

## OS
* Es una librería para poder acceder al sistema y poder interacturar con el desde nuestro .py
* **Importación:** import os

## RE
* Es una librería para encontrar patrones
* **Importación:** import re


# Instalación de TensorFlow y Keras 🔧
* Tenemos que abrir la terminal de **Anaconda Prompt como admin**
* Introducimos el siguiente comando 

```
conda install -c conda-forge keras
```

* Introducimos el siguiente fragmento de codigo en nuestro py y sino salta error estaría funcional:

```
# Importamos el modelo para crear las capas 
from keras.model import Sequential

# Creamos las capas de nuestra red neuronal
model = Sequential()
```
Nos deberia de sacar por pantalla
**Using TensorFlow backend**


# Acrónimos📋
* **CNN**: Redes Neuronales Convolucionales ( Convolutional Neural Network)

# Bibliografía 📖
_Mención y ayuda para recordar los conocimientos y sitios donde de ha encontrado al información_
* [Aprendiendo a manejar tensorflow y Keras](https://www.aprendemachinelearning.com/una-sencilla-red-neuronal-en-python-con-keras-y-tensorflow/) -  con un ejemplo

* [Red Neuronal convolucional](https://www.aprendemachinelearning.com/como-funcionan-las-convolutional-neural-networks-vision-por-ordenador/?utm_source=github&utm_medium=readme&utm_campaign=repositorio)- Denominado CNN

* [Mayor documentacion de redes neuronales](https://github.com/jbagnato/machine-learning) - Ejemplos y aumento de la bibliogrfía 

* [Consejos de mejorar de Deep Learning](https://www.aprendemachinelearning.com/12-consejos-utiles-para-aplicar-machine-learning/?utm_source=github&utm_medium=readme&utm_campaign=repositorio) - 
 Consejos útiles para el Machine Learning

* [Clasificación de Imágenes de deportes](https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/?utm_source=github&utm_medium=readme&utm_campaign=repositorio) - Ejemplo de  Convolutional Neural Network

* [Deep Learning](https://www.aprendemachinelearning.com/aprendizaje-profundo-una-guia-rapida/)- Ejemplo de Deep Learning

* [Ejemplo de red convolucional](https://codelabs.developers.google.com/codelabs/tensorflow-lab3-convolutions/#1)- Google convolution

* [Plantilla README.md](https://gist.github.com/Villanuevand/6386899f70346d4580c723232524d35a#ejecutando-las-pruebas-%EF%B8%8F) - Ejemplo de README.md

* [Estilo Yolo -](https://www.youtube.com/watch?v=SJRP0IRfPj0)- Parte 1

* [Estilo Yolo 2](https://www.youtube.com/watch?v=EKe05rMG-Ww) - Parte 2

* [Clasificador de imágenes](https://www.youtube.com/watch?v=FWz0N4FFL0U) - Con TensorFlow

* [Crear dataset](https://machinelearningmastery.com/how-to-load-large-datasets-from-directories-for-deep-learning-with-keras/)- ImageDataGen

## Documentación
* [Libreria Keras](https://keras.io/api/) - Información sobre la librería