#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:28:16 2021

@author: yani
"""

import random
import os
import torch
import torchvision.transforms as transforms
import skimage.util
import numpy as np


from PIL import Image #, ImageEnhance




class AddGaussianNoise:
    """Clase para añadir ruido gausiano"""

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# def gaussian(ins, is_training, mean, stddev):
#     if is_training:
#         noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
#         return ins + noise
#     return ins


class RandomCrop(object):
    """Clase para hacer un recorte con unos valores aleatorios dentro de limitaciones
    impuestas por el usuario"""

    def __init__(self, max_x, max_y):
        self.max_x = max_x
        self.max_y = max_y
        
    def __call__(self, tensor):
        x1 = random.randint(0, self.max_x)
        x2 = random.randint(self.max_x + 20, 60)
        y1 = random.randint(0, self.max_y)
        y2 = random.randint(self.max_y + 10,50)
        tensor[0, y1:y2, x1:x2] = 0
        return tensor

class RandomCropDerecha(object):
    """Clase para hacer el recorte en la parte derecha de la imagen"""
    def __init__(self, max_x, max_y):
        self.max_x = max_x
        self.max_y = max_y
    
    def __call__(self,tensor,posicion):

        if posicion=='derecha':
            x1 = random.randint(50, 50 + self.max_x)
            x2 = random.randint(50 + self.max_x, 100)

        elif posicion=='izquierda':
            x1 = random.randint(0, self.max_x)
            x2 = random.randint(self.max_x + 20, 60)
        
        y1 = random.randint(0, self.max_y)
        y2 = random.randint(self.max_y + 10,50)
        tensor[0, y1:y2, x1:x2] = 0

        return tensor


class RuidoScikit(object):
    """Clase para aplicar ruido gausiano mediante sci-kit."""
    def __init__(self,mode,seed):
        self.mode = mode
        self.seed = seed

    def __call__(self, image):
        image = np.asarray(image)
        image = skimage.util.random_noise(image, mode=self.mode , seed=self.seed)
        return image
    

trans_T = transforms.ToTensor()
trans_P = transforms.ToPILImage()
trans_Crop = RandomCrop(20, 20)


directorio_salida = input("Directorio donde se encuentran las imagenes a transformar : ")
directorio_guardar = input("Directorio donde guardamos las imagenes :")

directorios = [x[1] for x in os.walk(directorio_salida)]
directorios = directorios[0]

numeroImagen = 0

for directorio in directorios:
    medias = []
    desviaciones = []
    for i in os.listdir(directorio_salida + directorio):
        nombre_imagen = os.fsdecode(i)
        imagen = Image.open(directorio_salida + directorio + "/" + nombre_imagen)
        imagen = trans_T(imagen)
        mean_c1 = imagen[0, :, :].mean()
        std_c1 = imagen[0, :, :].std()
        medias.append(mean_c1)
        desviaciones.append(std_c1)
        trans_P(imagen)


    mean_c1 = sum(medias) / len(medias)
    std_c1 = sum(desviaciones) / len(desviaciones)    
        
    print("La media es :", mean_c1, "La desviacion estandar es:", std_c1)

    transformCropRuido=transforms.Compose([ 
        transforms.Resize((50, 100)),
        #transforms.Normalize((0.1,), (0.3,)),
        #transforms.ColorJitter(contrast = 0.7),
        transforms.ToTensor(),
        #RandomCrop(20, 20),
        transforms.Normalize((mean_c1,), (std_c1,)),
        #AddGaussianNoise(0.3 , 0.04 )
        #AddGaussianNoise(mean_c1, std_c1 ),
        transforms.ToPILImage(),

    ])

    transformCrop = transforms.Compose([
        transforms.Resize((50, 100)),
        transforms.ToTensor(),
        #RandomCrop(20, 20),
        transforms.ToPILImage()
    ])

    for i in os.listdir(directorio_salida + directorio):
        nombre_imagen = os.fsdecode(i)
        imagen = Image.open(directorio_salida + directorio  + "/" + nombre_imagen)
        #imagen = Image.open(directorio_salida + nombre_imagen)
        imagen_Crop = imagen
        imagen_Crop = transformCrop(imagen)
        imagen_Crop_Ruido = transformCropRuido(imagen)
        if not os.path.exists(directorio_guardar + directorio + "/"):
            os.makedirs(directorio_guardar + directorio + "/")
        #imagen_Crop.save(directorio_guardar + directorio + "/" + "crop" + str(numeroImagen) + nombre_imagen )
        #imagen_Crop_Ruido.save(directorio_guardar + directorio + "/" + "crop_ruido" + str(numeroImagen) + nombre_imagen )
        imagen_Crop.save(directorio_guardar + directorio + "/" + nombre_imagen )
        imagen_Crop_Ruido.save(directorio_guardar + directorio + "/" + "Ruido" + nombre_imagen )
        numeroImagen = numeroImagen + 1

"""
    imagen_negra = torch.zeros(1, 50, 100)
    imagen_blanca = torch.zeros(1, 50, 100)

    imagen_blanca[0, :, :] = 1

    imagen_negra = trans_P(imagen_negra)
    imagen_blanca = trans_P(imagen_blanca)

    imagen_blanca.save(directorio_guardar + directorio + "/" + "imagenBlanca.jpg")
    imagen_negra.save(directorio_guardar + directorio + "/" + "imagenNegra.jpg")
"""

