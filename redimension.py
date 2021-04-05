from PIL import Image
import os
import sys
import re
import numpy as np

'''
Funcion para obtener la ruta de las imagenes
'''
def getImagesPath(ruta):
    dirname = os.path.join(os.getcwd(), ruta)
    
    return dirname + os.sep

'''
Funcion para cambiar las dimensiones de la imagen
'''
def resize_image(ruta_origen, imagen, tam, ext):
    partes = ruta_origen.split("/")

    partes_2 = partes[1].replace("\\", "/")
    partes_2 = partes_2.split("/")
    # print(partes_2)
    img = Image.open(ruta_origen)
    new_img = img.resize((tam['ancho'], tam['alto']))
    ruta_dest = getImagesPath("TFG/procesadas/" + str(partes_2[0]) + "/" + str(partes_2[1]))
    new_img.save(ruta_dest + imagen + "." + str(ext) )

'''
Función para leer las imagenes que vamos a procesar
'''
def readImagesToResize(images_path, tam):
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
                    # print(filepath)
                    ext = filename.split(".")
                    resize_image(filepath, ext[0], tam, ext[1])
                    # nuevo_nombre = os.path.join(root, "DJI_" + cant +".JPG")
                    # os.rename(filepath, nuevo_nombre)
                    # b = "Leyendo..." + str(cant)
                    # print (b, end="\r")
                    if prevRoot != root:
                        print(root, cant)
                        prevRoot = root
                        directories.append(root)
                        dircount.append(cant)
                        cant = 0

path = getImagesPath("TFG/test")
readImagesToResize(path, {"ancho": 150, "alto": 150})