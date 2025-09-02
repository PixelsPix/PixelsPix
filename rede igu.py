# Bloco 1 - importação de bibliotecas e definição dos dados
import numpy as np
import matplotlib.pyplot as plt

# Dados do experimento: TRS, WS, e 5 propriedades medidas
dados = [
    [500, 6.25, 112, 86, 3.04, 27.2, 27.6],
    [500, 10, 104, 80, 3.34, 27.5, 28],
    [500, 16, 103, 78, 3.71, 28, 28.3],
    [500, 20, 102, 76, 4.78, 28.2, 28.5],
    [800, 6.25, 102, 77, 5.22, 26.8, 26.55],
    [800, 10, 101, 75, 5.44, 27, 26.6],
    [800, 16, 92, 66, 5.52, 27.6, 27],
    [800, 20, 91, 65, 5.63, 27.8, 27.6],
    [1000, 6.25, 99, 75, 5.65, 26.4, 26],
    [1000, 10, 91, 66, 6.36, 26.8, 24.8],
    [1000, 16, 89, 63, 6.94, 27, 26.7],
    [1000, 20, 88, 60, 7.63, 27.2, 27.2],
    [1250, 6.25, 88, 60, 7, 24, 24.1],
    [1250, 10, 87, 57, 7.02, 24.8, 24.8],
    [1250, 16, 86, 56, 7.36, 25.7, 25.3],
    [1250, 20, 84, 53, 8.36, 26.1, 25.9],
    [1500, 6.25, 80, 58, 7.45, 21.4, 23.4],
    [1500, 10, 79, 56, 10.17, 22, 23.7],
    [1500, 16, 77, 52, 10.39, 22.9, 24.8],
    [1500, 20, 76, 47, 11.26, 23.8, 25.2]
]
dados = np.array(dados)

# Bloco 2 - normalização dos dados e separação de entrada/saída
def leaky_ReLU(x):
    return np.where(x > 0, x, 0.01 * x)

def leaky_ReLU_derivada(x):
    return np.where(x > 0, 1, 0.01)

entrada = dados[:, 0:2]  # TRS e WS
saida = dados[:, 2:]     # 5 propriedades

def normalizar(data):
    return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

def desnormalizar(data_norm, data_original):
    return data_norm * (data_original.max(axis=0) - data_original.min(axis=0)) + data_original.min(axis=0)

entrada_norm = normalizar(entrada)
saida_norm = normalizar(saida)

# Bloco 3 - separação dos dados em treino e validação
np.random.seed(45)
idx = np.arange(len(entrada_norm))
np.random.shuffle(idx)

ntreino = 15
itrain, ival = idx[:ntreino], idx[ntreino:]

Xtr, Ytr = entrada_norm[itrain], saida_norm[itrain]
Xval, Yval = entrada_norm[ival],   saida_norm[ival]

in_min, in_max = entrada.min(axis=0), entrada.max(axis=0)
out_min, out_max = saida.min(axis=0), saida.max(axis=0)

# Bloco 4 - definição da arquitetura da rede e inicialização dos pesos
n_in, n_hidden, n_out = 2, 10, 5
W1 = np.random.uniform(-1, 1, size=(n_hidden, n_in+1))
W2 = np.random.uniform(-1, 1, size=(n_out,   n_hidden+1))

def add_bias_column(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])

# Bloco 5 - forward pass da rede neural
def forward(X, W1, W2):
    Xb = add_bias_column(X)
    Z1 = Xb @ W1.T
    A1 = leaky_ReLU(Z1)
    A1b = add_bias_column(A1)
    Z2 = A1b @ W2.T
    Yhat = leaky_ReLU(Z2)
    cache = (Xb, Z1, A1, A1b, Z2, Yhat)
    return Yhat, cache

def sse_batch(Yhat, Y):
    return 0.5*np.sum((Yhat - Y)**2, axis=1)

def mse_set(Yhat, Y):
    return np.mean(sse_batch(Yhat, Y))

# Bloco 6 - backpropagation com SGD amostra a amostra
lr = 0.02

def train_one_sample(x, t, W1, W2):
    yhat, (Xb, Z1, A1, A1b, Z2, Yhat) = forward(x[None, :], W1, W2)
    e = (yhat - t[None, :])
    delta2 = e * leaky_ReLU_derivada(Z2)
    delta1 = (delta2 @ W2[:, :-1]) * leaky_ReLU_derivada(Z1)
    dW2 = delta2.T @ A1b
    dW1 = delta1.T @ Xb
    W2 = W2 - lr * dW2
    W1 = W1 - lr * dW1
    return W1, W2

# Bloco 7 - treinamento da rede e monitoramento do erro
epochs = 5000
hist_mse_tr, hist_mse_val = [], []
best_W1, best_W2, best_mse_val, best_epoch = None, None, np.inf, -1

for ep in range(1, epochs+1):
    perm = np.random.permutation(len(Xtr))
    for i in perm:
        W1, W2 = train_one_sample(Xtr[i], Ytr[i], W1, W2)

    ytr, _  = forward(Xtr, W1, W2)
    yval, _ = forward(Xval, W1, W2)
    mse_tr  = mse_set(ytr, Ytr)
    mse_v   = mse_set(yval, Yval)
    hist_mse_tr.append(mse_tr)
    hist_mse_val.append(mse_v)

    if mse_v < best_mse_val:
        best_mse_val, best_epoch = mse_v, ep
        best_W1, best_W2 = W1.copy(), W2.copy()

# Bloco 8 - avaliação da rede com os melhores pesos e cálculo de R²
W1, W2 = best_W1, best_W2

def desnorm_preds(Yn):
    return Yn*(out_max - out_min) + out_min

yval_norm, _ = forward(Xval, W1, W2)
yval_real = desnorm_preds(yval_norm)
Yval_real = desnorm_preds(Yval)

nomes_saidas = [
    "Resistência à tração (MPa)",
    "Alongamento (%)",
    "Dureza (HV)",
    "Temperatura máxima (°C)",
    "Temperatura média (°C)"
]

print("R² para cada propriedade prevista:")
R2_list = []
for i in range(5):
    y = Yval_real[:, i]
    yhat = yval_real[:, i]
    SQT = np.sum((y - y.mean())**2)
    SQE = np.sum((yhat - y)**2)
    R2_i = 1 - SQE/SQT
    R2_list.append(R2_i)
    print(f"{nomes_saidas[i]}: R² = {R2_i:.4f}")

# Bloco 9 - função de predição para novos valores de TRS e WS
def prever(TZS_rpm, WS_mm_min):
    x = np.array([[TZS_rpm, WS_mm_min]])
    x_norm = (x - in_min) / (in_max - in_min)
    y_norm, _ = forward(x_norm, W1, W2)
    y_real = desnorm_preds(y_norm)
    return y_real.ravel()  # devolve 5 propriedades no domínio real

# Bloco 10 - visualização dos resultados

from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# 10.1 - Gráfico de erro MSE por época (treino vs. validação)
plt.figure(figsize=(8, 5))
plt.plot(hist_mse_tr, label='Erro de Treinamento', color='blue', linewidth=1.5)
plt.plot(hist_mse_val, label='Erro de Validação', color='red', linestyle='--', linewidth=1.5)
plt.xlabel('Épocas', fontsize=12)
plt.ylabel('Erro MSE', fontsize=12)
plt.title('Evolução do Erro MSE durante o Treinamento', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 10.2 - Gráficos de dispersão: valores reais vs. previstos
plt.figure(figsize=(15, 10))
for i in range(5):
    y_real = Yval_real[:, i]
    y_pred = yval_real[:, i]
    
    plt.subplot(2, 3, i+1)
    plt.scatter(y_real, y_pred, color='blue', s=60, label='Previsto')
    plt.plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'k--', label='Ideal')
    plt.xlabel('Valor Medido', fontsize=10)
    plt.ylabel('Valor Previsto', fontsize=10)
    plt.title(nomes_saidas[i], fontsize=12)
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()

# 10.3 - Gráficos 3D de superfície para cada propriedade prevista
fig = plt.figure(figsize=(18, 10))
for i in range(5):
    ax = fig.add_subplot(2, 3, i+1, projection='3d')

    # Dados de entrada reais
    x = entrada[ival, 0]  # TRS
    y = entrada[ival, 1]  # WS
    z = yval_real[:, i]   # propriedade prevista

    # Malha para interpolação
    xi = np.linspace(x.min(), x.max(), 30)
    yi = np.linspace(y.min(), y.max(), 30)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='cubic')

    # Superfície
    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', alpha=0.7, edgecolor='none')
    ax.scatter(x, y, z, color='red', s=40, label='Pontos previstos')

    ax.set_xlabel('TRS (rpm)', fontsize=10)
    ax.set_ylabel('WS (mm/min)', fontsize=10)
    ax.set_zlabel(nomes_saidas[i], fontsize=10)
    ax.set_title(f'{nomes_saidas[i]} (Superfície 3D)', fontsize=12)
    ax.legend()

plt.tight_layout()
plt.show()