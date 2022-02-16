#Import de libs
import cv2
import numpy as np
import imageio as iio

# Ejercicio 3: Escribe una función que aplique la normalización rg a una imagen.
def normalizaIm(imNp,markImg):

    # Ahora hago lo mismo con los datos normalizados segun la 'normalizacion rgb'
    #Básicamente normalizo(rgb)'/1(r+g+b)
    imrgbn = np.rollaxis((np.rollaxis(imNp,2)+0.0)/np.sum(imNp,2),0,3)[:,:,:2]# Divido el valor RGB de cada pixel por la suma de los tres valores
    return imrgbn#devuelvo los 2 primero valores que me interesa
    

