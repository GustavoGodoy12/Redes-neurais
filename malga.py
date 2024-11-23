# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 12:53:38 2021

@author: Prof Malga 

EXEMPLO DE UTILIZAÇÃO DE ARQUIVOS CSv

UTILIZEI O ARQUIVOS IRIS QUE TRATA-SE DE UM DATASET SOBRE CARACTERÍSTICAS 
DA FLOR IRIS 

VOCÊ NÃO PODERÁ REALIZAR SEU TRABALHO COM BASE NESSE DATASET.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Definindo o número de épocas e o num de amostras
numEpocas = 1000
q = 100

# Setosa 1 && Versicolor -1
#Aqui é feito a leitura de cada atributo descrito no arquivo CSV
#USECOLS=[] significa a coluna que da qual você está extraindo os dados.
CompSepala = pd.read_csv('Iris.csv', usecols=[1])
CompSepala = np.array(CompSepala)
CompSepala = np.transpose(CompSepala)

LargSepala = pd.read_csv('Iris.csv', usecols=[2])
LargSepala = np.array(LargSepala)
LargSepala = np.transpose(LargSepala)

CompPetala = pd.read_csv('Iris.csv', usecols=[3])
CompPetala = np.array(CompPetala)
CompPetala = np.transpose(CompPetala)

LargPetala = pd.read_csv('Iris.csv', usecols=[4])
LargPetala = np.array(LargPetala)
LargPetala = np.transpose(LargPetala)

# Bias
bias = 1

# Entrada do perceptron
X = np.vstack((CompSepala, LargSepala, CompPetala, LargPetala))
# Saída
Y = np.zeros(100)
for i in range(50):
    Y[i] = 1
for i in range(50,100):
    Y[i] = -1
    
# Taxa de aprendizagem
eta = 0.1

# Define o vetor de pesos
W = np.zeros([1, 5]) # Duas entradas + bias

# Armazenando erros
e = np.zeros(100)

def funcaoAtivacao(valor):
    if valor < 0.0:
        return -1
    else:
        return 1
    
for j in range(numEpocas):
    for k in range(q):
        # Insere bias no vetor de entrada
        Xb = np.hstack((bias, X[:,k]))
        
        # Calcula o campo induzido
        V = np.dot(W, Xb)
        
        # Calcular a saída do perceptron
        Yr = funcaoAtivacao(V)
        
        # Calcular o erro: e = (Y - Yr)
        e[k] = Y[k] - Yr
        
        # Treinamento de perceptron
        W = W + eta*e[k]*Xb
        
        #print(W)
        
plt.title("SEPALA")
plt.xlabel("Comprimento")
plt.ylabel("Largura")
plt.scatter(CompSepala,LargSepala)
plt.grid()
plt.show()

plt.title("PETELA")
plt.xlabel("Comprimento")
plt.ylabel("Largura")
plt.scatter(CompPetala,LargPetala)
plt.grid()
plt.show()
        
print("(e) = " + str(e))