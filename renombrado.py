# -*- coding: utf-8 -*-

import os
import re
import numpy as np

'''
Funcion para obtener la ruta de las imagenes
'''
def getImagesPath(ruta):
    dirname = os.path.join(os.getcwd(), ruta)
    
    return dirname + os.sep


'''
Función para leer las imagenes que vamos a procesar
'''
def readImagesToProcess(images_path):
    print("Lectura de imagenes de la ruta : " + images_path)
    #Modificar para obtener los datos
    images = []
    directories = []
    dircount = []
    prevRoot = ''
    cant = 0

    print("Vamos a leer las imagenes de ",images_path)

    for root, dirnames, filenames in os.walk(images_path):
        if(images_path != root):
            for filename in filenames:
                if re.search("\.(jpg|jpeg|png|bmp|tiff|JPG)$", filename):
                    cant = cant + 1
                    filepath = os.path.join(root, filename)
                    nuevo_nombre = os.path.join(root, "DJI_SC_" + str(cant) +".JPG")
                    os.rename(filepath, nuevo_nombre)
                    b = "Leyendo..." + str(cant)
                    print (b, end="\r")
                    if prevRoot != root:
                        print(root, cant)
                        prevRoot = root
                        directories.append(root)
                        dircount.append(cant)
readImagesToProcess(getImagesPath("TFG/procesadas/Sin cruces auto"))
