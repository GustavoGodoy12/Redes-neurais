import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dados_carro = pd.read_csv('CarPrice_Assignment.csv')

peso_carro = np.array(dados_carro["curbweight"][:100])
tamanho_motor = np.array(dados_carro["enginesize"][:100])
potencia_motor = np.array(dados_carro["horsepower"][:100])

X = np.vstack((peso_carro, tamanho_motor, potencia_motor))

escalador_X = StandardScaler()
X = escalador_X.fit_transform(X.T).T

Y = np.array(dados_carro["price"][:100])

escalador_Y = StandardScaler()
Y = escalador_Y.fit_transform(Y.reshape(-1, 1)).flatten()

eta = 0.1
numEpocas = 1000

tamanho_entrada = X.shape[0]
tamanhos_camadas_ocultas = [5, 4, 3]
tamanho_saida = 1

pesos = [np.random.randn(tamanho_entrada, tamanhos_camadas_ocultas[0]) * np.sqrt(2 / tamanho_entrada)]
for i in range(1, len(tamanhos_camadas_ocultas)):
    pesos.append(np.random.randn(tamanhos_camadas_ocultas[i-1], tamanhos_camadas_ocultas[i]) * np.sqrt(2 / tamanhos_camadas_ocultas[i-1]))
pesos.append(np.random.randn(tamanhos_camadas_ocultas[-1], tamanho_saida) * np.sqrt(2 / tamanhos_camadas_ocultas[-1]))

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def leaky_relu_derivada(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def feedforward(X, pesos):
    ativacoes = [X]
    for w in pesos[:-1]: 
        ativacoes.append(leaky_relu(np.dot(ativacoes[-1], w))) 
    ativacoes.append(np.dot(ativacoes[-1], pesos[-1]))  
    return ativacoes

def prever(X, pesos):
    ativacoes = feedforward(X, pesos)
    return ativacoes[-1]

X_treino, X_teste, Y_treino, Y_teste = train_test_split(X.T, Y, test_size=0.2, random_state=42)
Y_treino = Y_treino.reshape(-1, 1)
Y_teste = Y_teste.reshape(-1, 1)

def retropropagar(X, Y, pesos, ativacoes, taxa_aprendizado):
    m = X.shape[0]
    erros = [ativacoes[-1] - Y]  
    deltas = [erros[-1]]

    for i in range(len(pesos) - 2, -1, -1):
        delta = np.dot(deltas[0], pesos[i + 1].T) * leaky_relu_derivada(ativacoes[i + 1])
        deltas.insert(0, delta)

    for i in range(len(pesos)):
        pesos[i] -= taxa_aprendizado * np.dot(ativacoes[i].T, deltas[i]) / m  
    return pesos

def treinar(X, Y, pesos, epocas, taxa_aprendizado):
    historico_erro = []
    for epoca in range(epocas):
        ativacoes = feedforward(X, pesos)
        pesos = retropropagar(X, Y, pesos, ativacoes, taxa_aprendizado)
        erro = np.mean((ativacoes[-1] - Y) ** 2)  
        historico_erro.append(erro)
        if epoca % 100 == 0:
            print(f"Época {epoca}, Erro: {erro:.4f}")
    return pesos, historico_erro

pesos, historico_erro = treinar(X_treino, Y_treino, pesos, numEpocas, eta)

previsoes = prever(X_teste, pesos)

previsoes_originais = escalador_Y.inverse_transform(previsoes).flatten()
Y_teste_original = escalador_Y.inverse_transform(Y_teste).flatten()

print("Previsão do preço do carro: ", [f"{x:.2f}" for x in previsoes_originais[:10]])

plt.scatter(Y_teste_original, previsoes_originais)
plt.xlabel('Valor Real')
plt.ylabel('Valor Previsto')
plt.title('Previsão do Preço do Carro')
plt.grid(True)
plt.show()
