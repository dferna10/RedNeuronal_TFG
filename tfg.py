'''
Red Neuronal Convolucional (CNN)
@version 1.0
Con set de imagenes cargadas de poco en poco
'''
# Importamos las librerias necesarias para crear nuestra red
import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras.utils import plot_model 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model # Para cargar el modelo de nuestra red
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ModelCheckpoint #Para coger el mejor resultado de todas las epocas

# Importamos las librerias de actuacion contra el sistema
import os
import numpy as np
import matplotlib.pyplot as plt

import time

'''
*******************************************************************
*  Creación del modelo de la red neuronal convolucional           *
*******************************************************************
'''

'''
Funcion para crear el modelo de capas de nuestra red neuronal
'''
def createNeuralModel( n_clases, tam):
    
    modelo = Sequential()
    
    # Capa de entrada
    modelo.add(Conv2D(32, kernel_size = (3, 3), activation = "relu", padding='same', strides = 2, input_shape = (tam['alto'], tam['ancho'], 3)))
    modelo.add(BatchNormalization())
    modelo.add(MaxPooling2D(pool_size = (2, 2), strides = 2 , padding='same'))

    # Capas intermedias 
    modelo.add(Conv2D(64, kernel_size = (3, 3), activation = "relu", padding='same', strides = 2))
    modelo.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
    
    modelo.add(Conv2D(128, kernel_size = (3, 3), activation = "relu"))
    modelo.add(BatchNormalization())
    modelo.add(MaxPooling2D(pool_size = (2, 2), strides = 2 ))
    
    # modelo.add(Conv2D(128, kernel_size = (3, 3),  activation = "relu"))
    # modelo.add(MaxPooling2D(pool_size = (2, 2), strides = 1 ))
    
    # modelo.add(Conv2D(128, kernel_size = (3, 3),  activation = "relu"))
    # modelo.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Quitamos dimensionalidad a la imagen
    modelo.add(Flatten())

    modelo.add(Dense(1024, activation = "relu"))
    # En cada iteracion desactivamos el 50% de las neuronas para darle varios caminos y poder  mejorar
    modelo.add(Dropout(0.5)) 
    
    # Capa de salida
    modelo.add(Dense(n_clases, activation = 'softmax'))
    
    return modelo

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
    
    return dirname + os.sep 

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
Funcion para obtener los datos estadisticos del entrenamiento de nuestra red
para asi saber donde mejorar
Acurracy, loss
'''
def get_stats(modelo_entrenado):
    accuracy = modelo_entrenado.history['accuracy']
    val_accuracy = modelo_entrenado.history['val_accuracy']
    loss = modelo_entrenado.history['loss']
    val_loss = modelo_entrenado.history['val_loss']
    
    epochs = range(len(accuracy))
    plt.show()
    
    print("\n")
    
    plt.plot(epochs, accuracy, color = 'b', label = 'Accuracy entrenamiento')
    plt.plot(epochs, val_accuracy, color = 'r', label = 'Accuracy validacion')
    plt.title('Accuracy entrenamiento y validacion')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, color = "b", label='Loss entrenamiento')
    plt.plot(epochs, val_loss, color = 'r', label='Loss validacion')
    plt.title('Loss entrenamiento y validacion')
    plt.legend()
    plt.show()


'''
Funcion para imprimir la grafica cuando solo entrenamos sin validacion
'''
def getSimplyStats(modelo_entrenado):
    accuracy = modelo_entrenado.history['accuracy']
    loss = modelo_entrenado.history['loss']
    
    epochs = range(len(accuracy))
    plt.show()
    
    print("\n")
    
    plt.plot(epochs, accuracy, color = 'b', label = 'Accuracy')
    plt.plot(epochs, loss, color = "r", label='Loss')
    plt.title('Acuraccy y loss entrenamiento')
    plt.legend()
    plt.show()

'''
Funcion para registrar los resultados obtenidos al entrenar nuestra red
'''
def write_simply_log(epochs, accuracy,loss, loss_test, accuracy_test):
    actual = time.strftime("%c")
    f = open ("train_log.txt", "a")
    cadena = "\n" + actual
    cadena = cadena + "\nNúmero de epocas: " + str(epochs)
    cadena = cadena + "\nAccuracy: " + str(list(accuracy))
    cadena = cadena + "\nLoss : " + str(list(loss))
    cadena = cadena + "\nAccuracy Test : " + str(accuracy_test)
    cadena = cadena + "\nLoss test : " + str(loss_test)
    cadena = cadena + "\n"
    f.write(cadena)
    f.close()

'''
Funcion para registrar los resultados obtenidos al entrenar nuestra red
'''
def write_log(epochs, accuracy, accuracy_validation, loss, loss_validacion, loss_test, accuracy_test):
    actual = time.strftime("%c")
    f = open ("train_log.txt", "a")
    cadena = "\n" + actual
    cadena = cadena + "\nNúmero de epocas: " + str(epochs)
    cadena = cadena + "\nAccuracy: " + str(list(accuracy))
    cadena = cadena + "\nAccuracy validacion : " + str(list(accuracy_validation))
    cadena = cadena + "\nLoss : " + str(list(loss))
    cadena = cadena + "\nLoss validacion : " + str(list(loss_validacion))
    cadena = cadena + "\nAccuracy Test : " + str(accuracy_test)
    cadena = cadena + "\nLoss test : " + str(loss_test)
    cadena = cadena + "\n"
    f.write(cadena)
    f.close()

'''
Funcion para guardar la red convolucional ya entrenada
'''
def save_cnn(modelo, ruta):
    print("\nGuardamos nuestro modelo")
    target_dir = create_directory(ruta)
    
    # Guaradamos nuestra red entrenada
    modelo.save(target_dir + 'tfg.h5')
    # Guaradamos los pesos de nuestra red entrenada
    modelo.save_weights(target_dir + 'pesos.h5')

'''
Funcion principal para el entrenamiento de la red neuronal 
convoluciona (CNN)
'''
def train_cnn(rutas, tam):
    
    # Generamos las rutas
    ruta_entrenamiento = rutas['entrenamiento']
    # ruta_validacion = rutas['validacion']
    ruta_test = rutas['test']
    
    # Numero de clases o etiquetas de nuestra red
    clases = int(4)
    
    # Creamos el modelo de nuestra red neuronal
    modelo  = createNeuralModel(clases, tam)
    
    # Generador de imagenes para que se carguen cuando se necesiten
    entrena_datagen = ImageDataGenerator(
        rescale = 1. / 255,
        rotation_range = 40,
        horizontal_flip = True,
        vertical_flip = True,
    )
    
    datagen = ImageDataGenerator( rescale = 1. / 255)
    
    
    # Creamos las constantes de nuestra red
    batch_size = 15
    lr = 0.0001
    epochs = 100
    steps = 16
    # validation_steps = 350
    
    # Mostramos un resumen de nuestra red
    modelo.summary()
    # Capturamos la estructura de nuestra red
    plot_model(modelo, show_shapes=True, to_file="convolucional.png")
    # Compilamos la red
    modelo.compile(loss = 'categorical_crossentropy', 
                   optimizer = optimizers.Adam(learning_rate = lr), 
                   metrics = ['accuracy'])
    
    print("\n")
    # Creamos los sets de entrenamiento, validacion
    entrenamiento = entrena_datagen.flow_from_directory(directory = ruta_entrenamiento,
                                                        target_size = (tam['alto'], tam['ancho']),
                                                        class_mode = 'categorical',
                                                        batch_size = batch_size)
    
    # validacion = datagen.flow_from_directory(directory = ruta_validacion,
                                              # target_size = (tam['alto'], tam['ancho']),
                                              # class_mode = 'categorical',
                                              # batch_size = batch_size)

    print("\n")
   
    directorio = create_directory(rutas['modelo_temp'])
    #Creamos los checkpoints para que nos coja la mejor iteracion
    checkpoints = ModelCheckpoint(filepath = directorio + "mejores_pesos_{epoch:02d}_{accuracy:.2f}.hdf5", 
                                  monitor = 'accuracy', 
                                  verbose = 1, 
                                  save_best_only = True
                                  )
    
    # Entrenamos nuestra red
    modelo_entrenado = modelo.fit(entrenamiento,
                                   callbacks = [checkpoints],
                                  # steps_per_epoch = steps, 
                                  epochs = epochs, 
                                  # validation_data = validacion, 
                                  # validation_steps = validation_steps
                                  )
    
    print("\nEstadisticas de entrenamiento")
    getSimplyStats(modelo_entrenado)
    # get_stats(modelo_entrenado)
    
    # Guaradamos el modelo de nuestra red ya entrenada
    save_cnn(modelo, rutas["modelo"])
    
    # Cargamos las imágenes de test    
    test = datagen.flow_from_directory(directory = ruta_test,
                                        shuffle = False,
                                        target_size = (tam['alto'], tam['ancho']),
                                        class_mode = 'categorical',
                                        batch_size = batch_size)
    
    # Evaluamos nuestro modelo
    print("\nEvaluamos nuestro modelo")
    test_loss, test_accuracy = modelo.evaluate(test)

    print("\nEstadisticas de test\n")
    print('Test loss:', test_loss)
    print('Test accuracy:', test_accuracy)
    
    historico = modelo_entrenado.history
    
    write_simply_log(epochs, historico['accuracy'], historico['loss'],test_loss, test_accuracy)
    
    write_log(epochs, historico['accuracy'] , historico['val_accuracy'],
               historico['loss'] , historico['val_loss'],
               test_loss, test_accuracy)
  
    
  
'''
********************************************************************
* Funciones para la predicción de un nuevo elemento                *
********************************************************************
''' 
  
'''
Funcion para cargar nuestra red para poder predecir resultados
'''
def load_cnn(ruta):
    modelo_temp = ruta + "/mejores_pesos_95_0.99.hdf5"
    modelo_preentrenado =  modelo_temp
    pesos_modelo = modelo_temp
    # modelo_preentrenado = ruta + 'tfg.h5'
    # pesos_modelo = ruta + 'pesos.h5'
    
    modelo = None
    
    if(os.path.exists(modelo_preentrenado) and os.path.exists(pesos_modelo)):
        modelo = load_model(modelo_preentrenado)
        modelo.load_weights(pesos_modelo)
    
    return modelo  

'''
Funcion para predecir un nuevo elemento 
'''
def predict_element(rutas, imagen, tam):
    
    # modelo = load_cnn(rutas['modelo'])
    modelo = load_cnn(rutas['modelo_temp'])
    
    if(modelo != None):
        # test_cnn(modelo, rutas, tam, 15)
        imagen_test_in = load_img(rutas['prediccion'] +  imagen, target_size = (tam['alto'], tam['ancho']))
        imagen_test = img_to_array(imagen_test_in) 
        # imagen_test = imagen_test / 255.
        imagen_test = np.expand_dims(imagen_test, axis = 0)
        
        print(imagen_test.shape)
        
        prediccion = modelo.predict(imagen_test)
        salida = prediccion[0]
  
        print(salida)
  
        etiqueta = np.argmax(salida)
        etiqueta_salida = "Sin patron"
        
        if etiqueta == 0:
            print("Predicción: Canal")
            etiqueta_salida = "Canal"
        elif etiqueta == 1:
            print("Predicción: Cruce")
            etiqueta_salida = "Cruce"
        elif etiqueta == 2:
            print("Predicción: Mina")
            etiqueta_salida = "Mina"
        elif etiqueta == 3:
            print("Predicción: Sin elemento")
            etiqueta_salida = "Sin elemento"
            
        elif etiqueta == 4:
            print("Predicción: Sin patron")
            etiqueta_salida = "Sin patron"
        
        print(etiqueta_salida)
        plt.imshow(imagen_test_in, cmap='gray')
        plt.title("Clase {}". format(etiqueta_salida))
        
        return etiqueta_salida
         
    else:
        return -1

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

    tamanho = {
        "ancho": 150,
        "alto": 150
    }
    
    
    # Ruta de las imagenes sin procesar, de donde podemos extraer los directorios 
    # para obtener las etiquetas
    rutas = {
        'entrenamiento': getImagesPath("TFG/entrenamiento"),
        'validacion': getImagesPath("TFG/validacion"),
        'test': getImagesPath("TFG/test"),
        'modelo': getImagesPath("TFG/modelo"),
        'modelo_temp': getImagesPath("TFG/modelo_temp"),
        "prediccion":  getImagesPath("TFG/prediccion")
    }
    
    K.clear_session()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    
    if(opcion == 1):
        # Entrenamos nuestra red
        train_cnn(rutas, tamanho) 

    elif(opcion == 2):
        # Con cruce
        imagen = "DJJ_172.JPG" 
        # Sin cruce
        imagen = "DJJ_1583.JPG"
        imagen = "DJJ_408.JPG"
        # Mina
        imagen = "DJJ_861.JPG"
        
        # Canal
        imagen = "DJJ_121.JPG"
        imagen = "DJJ_140.JPG"
        
        
        #Otra imagen
        # imagen = "header.jpg"
        
        valor = predict_element(rutas, imagen, tamanho)
        print(valor)

        if(valor == -1):
            print("ERROR: No se encuentran los archivos de entrenamiento de la red")
            print("Por favor comuniqueselo al administrador del sistema")

        return valor
    else:
        print("ERROR: No es una opción válida")
        show_menu()
        return -1

# import tensorflow as tf
# # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# print("Num GPUs Available: ", list(tf.config.experimental.list_physical_devices()))

# if __name__=='__main__':
main()  
        
#Para ver las GPU o dispositivos que tenemos
# import tensorflow as tf
# dispositivos = tf.config.experimental.list_physical_devices('GPU')

# print("Numero de GPU " +str(len(dispositivos)))
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())