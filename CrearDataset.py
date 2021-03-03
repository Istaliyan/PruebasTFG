#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:28:16 2021

@author: yani
"""

import random
import os
import torch
#import torchvision
#from IPython.display import display
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
#import numpy as np

#from torch.autograd import Variable
#import cv2
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
    


#pil_img = Image.open('./DatasetSoldaduras/NFD1/Pieza_15_Soldadura_77.png')
#pil_img = Image.open('./DatasetSoldaduras/NFD1/download.jpg').convert('RGB')
trans_T = transforms.ToTensor()
# trans_S = transforms.Resize((200,1000))
trans_P = transforms.ToPILImage()
trans_Crop = RandomCrop(20, 20)
# trans_C = transforms.ColorJitter(contrast = 0.7)
#trans_P(trans_S(trans_T(pil_img))).show()


#pil_img = ImageEnhance.Contrast(pil_img).enhance(2)


# display(pil_img)

#torch.set_printoptions(edgeitems=3)
# display(trans_C(pil_img))
# pil_img = trans_T(pil_img)

# print(pil_img.size())
# print(pil_img.dim())


# x1 = random.randint(0, 40)
# x2 = random.randint(60, 80)
# y1 = random.randint(0, 50)
# y2 = random.randint(200,250)

# pil_img[0,x1:x2,y1:y2] = 0

# pil_img = trans_P(pil_img)

# display(pil_img)

# pil_img = trans_T(pil_img)

#print(torch.nonzero(pil_img))

directorio_salida = input("Directorio donde se encuentran las imagenes a transformar : ")
directorio_guardar = input("Directorio donde guardamos las imagenes :")

directorios = [x[1] for x in os.walk(directorio_salida)]

directorios = directorios[0]
#directory = os.fsencode(directorio_salida)

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
        
    # mean_c1 = pil_img[0, :, :].mean()
    # std_c1 = pil_img[0, :, :].std()
    print("La media es :", mean_c1, "La desviacion estandar es:", std_c1)

    transformCropRuido=transforms.Compose([ 
        transforms.Resize((50, 100)),
        #transforms.Normalize((0.1,), (0.3,)),
        #transforms.ColorJitter(contrast = 0.7),
        transforms.ToTensor(),
        RandomCrop(20, 20),
        transforms.Normalize((mean_c1,), (std_c1,)),
        #AddGaussianNoise(0.3 , 0.04 )
        AddGaussianNoise(mean_c1, std_c1 ),
        transforms.ToPILImage()
    ])

    transformCrop = transforms.Compose([
        transforms.Resize((50, 100)),
        transforms.ToTensor(),
        RandomCrop(20, 20),
        transforms.ToPILImage()
    ])

    #directorio_guardar = './DatasetSoldadurasMod/NFD1/'

    for i in os.listdir(directorio_salida + directorio):
        nombre_imagen = os.fsdecode(i)
        imagen = Image.open(directorio_salida + directorio  + "/" + nombre_imagen)
        #imagen = Image.open(directorio_salida + nombre_imagen)
        imagen_Crop = imagen
        imagen_Crop = transformCrop(imagen)
        imagen_Crop_Ruido = transformCropRuido(imagen)
        if not os.path.exists(directorio_guardar + directorio + "/"):
            os.makedirs(directorio_guardar + directorio + "/")
        imagen_Crop.save(directorio_guardar + directorio + "/" + "crop" + nombre_imagen )
        imagen_Crop_Ruido.save(directorio_guardar + directorio + "/" + "crop_ruido" + nombre_imagen)




    #Añadimos 1 imagen blanca y otra negra por completo

    imagen_negra = torch.zeros(1, 50, 100)
    imagen_blanca = torch.zeros(1, 50, 100)

    imagen_blanca[0, :, :] = 1

    imagen_negra = trans_P(imagen_negra)
    imagen_blanca = trans_P(imagen_blanca)

    imagen_blanca.save(directorio_guardar + directorio + "/" + "ImagenBlanca.jpg")
    imagen_negra.save(directorio_guardar + directorio + "/" + "ImagenNegra.jpg")



#pil_img = gaussian(trans_T(pil_img), True, 0.3, 0.04)
#print(pil_img)
#pil_img = trans_P(pil_img) 

#display(pil_img)
#pil_img.show()
#pil_img = trans_P(pil_img)




#print(pil_img.size())
# pil_img = trans_P(transform(pil_img))

# #pil_img = trans_S(pil_img)
# #print("La longitud del tensor es :" , pil_img.nelement())
# print(pil_img)
# display(pil_img)
#pil_img.show()


#img = cv2.imread('./DatasetSoldaduras/NFD1/Pieza_15_Soldadura_77.png')
#blur = cv2.GaussianBlur(img, (5,5), 0)
#plt.imshow(blur)
#cv2.imwrite('Gauss.png',blur)
