import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy

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

X_treino, X_teste, Y_treino, Y_teste = train_test_split(X.T, Y, test_size=0.2, random_state=42)
Y_treino = Y_treino.reshape(-1, 1)
Y_teste = Y_teste.reshape(-1, 1)


eta = 0.01  
numEpocas = 1000

tamanho_entrada = X.shape[0]
tamanhos_camadas_ocultas = [5, 4, 3]
tamanho_saida = 1

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def leaky_relu_derivada(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivada(x):
    s = sigmoid(x)
    return s * (1 - s)

def feedforward(X, pesos, ativacao):
    ativacoes = [X]
    for i, w in enumerate(pesos[:-1]):
        z = np.dot(ativacoes[-1], w)
        if ativacao == 'leaky_relu':
            a = leaky_relu(z)
        elif ativacao == 'sigmoid':
            a = sigmoid(z)
        ativacoes.append(a)
    ativacoes.append(np.dot(ativacoes[-1], pesos[-1]))
    return ativacoes

def prever(X, pesos, ativacao):
    ativacoes = feedforward(X, pesos, ativacao)
    return ativacoes[-1]

def retropropagar(X, Y, pesos, ativacoes, ativacao_func, taxa_aprendizado):
    m = X.shape[0]
    erros = ativacoes[-1] - Y  
    deltas = [erros]  

    for i in range(len(pesos)-2, -1, -1):
        if ativacao_func == 'leaky_relu':
            deriv = leaky_relu_derivada(ativacoes[i+1])
        elif ativacao_func == 'sigmoid':
            deriv = sigmoid_derivada(ativacoes[i+1])
        delta = np.dot(deltas[0], pesos[i+1].T) * deriv
        deltas.insert(0, delta)

    for i in range(len(pesos)):
        pesos[i] -= taxa_aprendizado * np.dot(ativacoes[i].T, deltas[i]) / m

    return pesos, erros  

def treinar(X, Y, pesos, epocas, taxa_aprendizado, ativacao_func):
    historico_erro = []
    for epoca in range(epocas):
        ativacoes = feedforward(X, pesos, ativacao_func)
        pesos, erros = retropropagar(X, Y, pesos, ativacoes, ativacao_func, taxa_aprendizado)
        erro = np.mean((ativacoes[-1] - Y) ** 2)
        historico_erro.append(erro)
        if epoca % 100 == 0:
            print(f"Época {epoca}, Erro: {erro:.4f}")
            print(f"Vetor de Erros: {erros.flatten()[:10]}...") 
    return pesos, historico_erro

def inicializar_pesos(tamanho_entrada, tamanhos_camadas_ocultas, tamanho_saida):
    pesos = [np.random.randn(tamanho_entrada, tamanhos_camadas_ocultas[0]) * np.sqrt(2 / tamanho_entrada)]
    for i in range(1, len(tamanhos_camadas_ocultas)):
        pesos.append(np.random.randn(tamanhos_camadas_ocultas[i-1], tamanhos_camadas_ocultas[i]) * np.sqrt(2 / tamanhos_camadas_ocultas[i-1]))
    pesos.append(np.random.randn(tamanhos_camadas_ocultas[-1], tamanho_saida) * np.sqrt(2 / tamanhos_camadas_ocultas[-1]))
    return pesos

print("Treinamento com Leaky ReLU:")
pesos_iniciais = inicializar_pesos(tamanho_entrada, tamanhos_camadas_ocultas, tamanho_saida)
pesos_leaky, historico_erro_leaky = treinar(X_treino, Y_treino, copy.deepcopy(pesos_iniciais), numEpocas, eta, 'leaky_relu')

print("\nTreinamento com Sigmoid:")
pesos_sigmoid, historico_erro_sigmoid = treinar(X_treino, Y_treino, copy.deepcopy(pesos_iniciais), numEpocas, eta, 'sigmoid')

previsoes_leaky = prever(X_teste, pesos_leaky, 'leaky_relu')
previsoes_sigmoid = prever(X_teste, pesos_sigmoid, 'sigmoid')

previsoes_leaky_originais = escalador_Y.inverse_transform(previsoes_leaky).flatten()
previsoes_sigmoid_originais = escalador_Y.inverse_transform(previsoes_sigmoid).flatten()
Y_teste_original = escalador_Y.inverse_transform(Y_teste).flatten()

print("\nPrevisões com Leaky ReLU:")
print([f"{x:.2f}" for x in previsoes_leaky_originais[:10]])

print("\nPrevisões com Sigmoid:")
print([f"{x:.2f}" for x in previsoes_sigmoid_originais[:10]])

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(Y_teste_original, previsoes_leaky_originais, color='blue', label='Previsões Leaky ReLU')
plt.plot([min(Y_teste_original), max(Y_teste_original)], [min(Y_teste_original), max(Y_teste_original)], 'r--', label='Ideal')
plt.xlabel('Valor Real')
plt.ylabel('Valor Previsto')
plt.title('Previsão do Preço do Carro - Leaky ReLU')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(Y_teste_original, previsoes_sigmoid_originais, color='green', label='Previsões Sigmoid')
plt.plot([min(Y_teste_original), max(Y_teste_original)], [min(Y_teste_original), max(Y_teste_original)], 'r--', label='Ideal')
plt.xlabel('Valor Real')
plt.ylabel('Valor Previsto')
plt.title('Previsão do Preço do Carro - Sigmoid')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(historico_erro_leaky, label='Leaky ReLU')
plt.plot(historico_erro_sigmoid, label='Sigmoid')
plt.xlabel('Épocas')
plt.ylabel('Erro Médio Quadrático')
plt.title('Histórico de Erros durante o Treinamento')
plt.legend()
plt.grid(True)
plt.show()
