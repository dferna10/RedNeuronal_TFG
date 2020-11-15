'''
Red Neuronal Convolucional (CNN)
@version 1.0
Con set de imagenes cargadas de poco en poco
'''
# Importamos las librerias necesarias para crear nuestra red
from keras import backend as K
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import load_model # Para cargar el modelo de nuestra red
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, Activation

# Importamos las librerias de actuacion contra el sistema
import os
import re


'''
*******************************************************************
*  Creación del modelo de la red neuronal convolucional           *
*******************************************************************
'''

'''
Funcion para crear el modelo de capas de nuestra red neuronal
'''
def createNeuralModel( n_categories, tam):
    model = Sequential()
    #Creamos la capa convolucional
    # Agregar parametros en el Conv2D
    # input_shape( Dimensiones de la imagens, colores)
    model.add(Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(tam['alto'], tam['ancho'], 3)))
    # Hacemos el pooling para recortar caracteristicas
    model.add(MaxPooling2D((2, 2),padding='same'))
    
    model.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
    # Hacemos el pooling para recortar caracteristicas
    model.add(MaxPooling2D((2, 2),padding='same'))
    
    model.add(Conv2D(128, kernel_size=(3,3), activation="sigmoid"))
    # Hacemos el pooling para recortar caracteristicas
    model.add(MaxPooling2D((2, 2),padding='same'))
    
    # Quitamos dimensionalidad a la imagen
    model.add(Flatten())
    
    model.add(Dense(256, activation="relu"))
    # model.add(Dense(64, activation="relu"))
    # En cada iteracion desactivamos el 50% de las neuronas para darle varios caminos y poder  mejorar
    model.add(Dropout(0.5))
    
    model.add(Dense(n_categories, activation='softmax'))

    return model


'''
*******************************************************************
*  Funciones comunes par el entrenamiento y prediccion            *
*******************************************************************
'''

'''
Funcion para obtener la ruta de las imagenes
'''
def getImagesPath(ruta):
    dirname = os.path.join(os.getcwd(), ruta)
    
    images_path = dirname + os.sep 
    
    return images_path

'''
Funcion para crear el directorio, si no existe
'''
def create_directory(ruta):
    target_dir =  os.path.join(os.getcwd(), ruta)
    target_dir = target_dir + os.sep
 
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
        
    return target_dir


'''
*******************************************************************
*  Funciones para el entrenamiento de nuestra red                 *
*******************************************************************
'''


'''
Funcion para guardar la red convolucional ya entrenada
'''
def save_cnn(modelo):
    print("\nGuardamos nuestro modelo")
    target_dir = create_directory("TFG/modelo")
    
    #Guaradamos nuestra red entrenada
    modelo.save(target_dir + 'tfg.h5')
    # Guaradamos los pesos de nuestra red entrenada
    modelo.save_weights(target_dir + 'pesos.h5')

'''
Funcion principal para el entrenamiento de la red neuronal 
convoluciona (CNN)
'''
def train_cnn(rutas, tam):
    
    # Generamos las rutas
    ruta_entrenamiento = getImagesPath(rutas['entrenamiento'])
    ruta_validacion = getImagesPath(rutas['validacion'])
    ruta_test = getImagesPath(rutas['test'])
    
    clases = 2
    
    #Creamos el modelo de nuestra red neuronal
    modelo  = createNeuralModel(clases, tam)
    
    # Generador de imagenes para que se carguen cuando se necesiten
    datagen = ImageDataGenerator()
    
    # Creamos las constantes de nuestra red
    batch_size = 32
    lr = 0.0004
    epochs = 20
    steps = 1000
    validation_steps = 300
    
    #Creamos los sets de entrenamiento, validacion y test
    entrenamiento = datagen.flow_from_directory(ruta_entrenamiento, target_size = (tam['alto'], tam['ancho']), class_mode = 'binary', batch_size = batch_size)
     
    validacion = datagen.flow_from_directory(ruta_validacion,  target_size = (tam['alto'], tam['ancho']), class_mode = 'binary', batch_size = batch_size)
    
    test = datagen.flow_from_directory(ruta_test,  target_size = (tam['alto'], tam['ancho']), class_mode = 'binary', batch_size = batch_size)

    # Compilamos la red
    modelo.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr = lr), metrics=['accuracy'])
    
    
    modelo.fit_generator(entrenamiento, steps_per_epoch = steps, epochs = epochs, validation_data = validacion, validation_steps = validation_steps)
    
    # Guaradamos el modelo de nuestra red ya entrenada
    save_cnn(modelo)
    
    

'''
********************************************************************
* Funcion principal del la red neuronal para nuestro mayor control *
********************************************************************
'''

'''
Funcion para imprimir el menu de nuestra red neuronal
'''
def show_menu():
    print("\nSelecciona alguna de nuestras opciones")
    print("1 - Entrenar nuestra red")
    print("2 - Comprobar elemento")
    print("\n")


'''
Funcion principal de la red neuronal para crear el control
'''
def main():
    opcion = 1
    
    K.clear_session()
    
    # tamanho = {
    #     "ancho": 150,
    #     "alto": 100
    # }
    tamanho = {
        "ancho": 400,
        "alto": 300
    }
    
    if(opcion == 1):
        # Ruta de las imagenes sin procesar, de donde podemos extraer los directorios 
        # para obtener las etiquetas
        rutas = {
            'entrenamiento': "TFG/entrenamiento",
            'validacion': "TFG/validacion",
            'test': "TFG/test"
        }
        
        # Entrenamos nuestra red
        train_cnn(rutas, tamanho) 

    elif(opcion == 2):
        imagen = "DJI_0021.JPG"
#         imagen = "camion.png"
        ruta_modelo = "TFG/modelo"
        ruta_prediccion = "TFG/test"
        
        # valor = clasify_element(ruta_modelo, ruta_prediccion, ruta_origen, imagen, tamanho)
        # if(valor == -1):
        #     print("ERROR: No se encuentran los archivos de entrenamiento de la red")
        #     print("Por favor comuniqueselo al administrador del sistema")
        
        # return valor
    else:
        print("ERROR: No es una opción válida")
        show_menu()
        return -1


main()    