"""modelos.py - Implementações dos modelos Perceptron, MLP e CNN.

- Perceptron: implementação didática em NumPy puro, com suporte a degrau,
  sigmoide e tanh, modo binário e multiclasse (One-vs-Rest).
- MLP: rede multicamadas configurável em PyTorch.
- CNN: rede convolucional simples em PyTorch (1 Conv2D + Pool + Linear).
Todos os modelos gravam o histórico época a época dos pesos, bias, perda,
acurácia e norma da variação dos parâmetros.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss


# ==============================================================================
# 1. PERCEPTRON EM NUMPY PURO
# ==============================================================================

class PerceptronNumPy:
    """Perceptron de camada única implementado com NumPy puro.

    Ativações suportadas:
      - 'degrau': regra de aprendizado clássica de Rosenblatt.
      - 'sigmoide': regra delta baseada no gradiente do erro.
      - 'tanh': regra delta adaptada para saída simétrica [-1, +1].
    Suporta classificação binária (1 neurônio) e multiclasse (One-vs-Rest).
    """

    def __init__(self, ativacao="sigmoide", taxa_aprendizado=0.1, epocas=50,
                 tamanho_lote=16, semente=42):
        self.ativacao = ativacao
        self.taxa = taxa_aprendizado
        self.epocas = epocas
        self.tamanho_lote = tamanho_lote
        self.semente = semente
        self.rng = np.random.RandomState(semente)

        self.W = None       # pesos: shape (d,) ou (K, d)
        self.b = None       # bias: float ou shape (K,)
        self.num_classes = 2
        self.historico = []

    def _ativar(self, z):
        """Aplica a função de ativação escalar ou matricial."""
        if self.ativacao == "degrau":
            return np.where(z >= 0.0, 1.0, 0.0)
        elif self.ativacao == "sigmoide":
            z_clipped = np.clip(z, -50.0, 50.0)
            return 1.0 / (1.0 + np.exp(-z_clipped))
        elif self.ativacao == "tanh":
            return np.tanh(z)
        else:
            raise ValueError(f"Ativação desconhecida: {self.ativacao}")

    def _derivada(self, y_ativado):
        """Derivada da função de ativação em relação a z (expressa em função de y)."""
        if self.ativacao == "degrau":
            return 1.0  # na regra clássica o erro multiplica diretamente o delta
        elif self.ativacao == "sigmoide":
            return y_ativado * (1.0 - y_ativado)
        elif self.ativacao == "tanh":
            return 1.0 - (y_ativado ** 2)
        return 1.0

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        N, d = X_train.shape
        classes_unicas = np.unique(y_train)
        self.num_classes = len(classes_unicas)
        multiclasse = self.num_classes > 2

        # Inicialização pequena aleatória (época 0)
        if multiclasse:
            K = self.num_classes
            self.W = self.rng.uniform(-0.1, 0.1, size=(K, d)).astype(np.float64)
            self.b = np.zeros(K, dtype=np.float64)
        else:
            self.W = self.rng.uniform(-0.1, 0.1, size=d).astype(np.float64)
            self.b = 0.0

        # Grava estado inicial (época 0)
        self.historico_pesos = [self.W.copy()]
        self.historico_bias = [self.b.copy() if multiclasse else float(self.b)]
        self.historico_perda = []
        self.historico_acc_treino = []
        self.historico_acc_teste = []
        self.normas_variacao = []

        # Métricas na época 0
        y_pred_tr = self.predict(X_train)
        self.historico_acc_treino.append(accuracy_score(y_train, y_pred_tr) * 100.0)
        if X_test is not None:
            y_pred_te = self.predict(X_test)
            self.historico_acc_teste.append(accuracy_score(y_test, y_pred_te) * 100.0)
        self.historico_perda.append(self._calcular_perda(X_train, y_train))

        # Loop de épocas de 1 até self.epocas
        indices = np.arange(N)
        for ep in range(1, self.epocas + 1):
            self.rng.shuffle(indices)
            W_anterior = self.W.copy()

            # Treinamento em minilotes
            for start in range(0, N, self.tamanho_lote):
                batch_idx = indices[start:start + self.tamanho_lote]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                if not multiclasse:
                    # Alvo binário
                    t_batch = y_batch.astype(np.float64)
                    if self.ativacao == "tanh":
                        t_batch = np.where(y_batch == 1, 1.0, -1.0)

                    z = np.dot(X_batch, self.W) + self.b
                    y_out = self._ativar(z)
                    erro = t_batch - y_out

                    if self.ativacao == "degrau":
                        # Regra clássica de Rosenblatt
                        dW = np.dot(erro, X_batch) / len(batch_idx)
                        db = np.sum(erro) / len(batch_idx)
                    else:
                        # Regra delta com gradiente
                        grad = erro * self._derivada(y_out)
                        dW = np.dot(grad, X_batch) / len(batch_idx)
                        db = np.sum(grad) / len(batch_idx)

                    self.W += self.taxa * dW
                    self.b += self.taxa * db
                else:
                    # One-vs-Rest para multiclasse
                    for k in range(self.num_classes):
                        t_k = (y_batch == k).astype(np.float64)
                        if self.ativacao == "tanh":
                            t_k = np.where(y_batch == k, 1.0, -1.0)

                        z_k = np.dot(X_batch, self.W[k]) + self.b[k]
                        y_out_k = self._ativar(z_k)
                        erro_k = t_k - y_out_k

                        if self.ativacao == "degrau":
                            dW_k = np.dot(erro_k, X_batch) / len(batch_idx)
                            db_k = np.sum(erro_k) / len(batch_idx)
                        else:
                            grad_k = erro_k * self._derivada(y_out_k)
                            dW_k = np.dot(grad_k, X_batch) / len(batch_idx)
                            db_k = np.sum(grad_k) / len(batch_idx)

                        self.W[k] += self.taxa * dW_k
                        self.b[k] += self.taxa * db_k

            # Norma da variação dos pesos nesta época
            norma_delta = float(np.linalg.norm(self.W - W_anterior))
            self.normas_variacao.append(norma_delta)

            # Histórico do estado
            self.historico_pesos.append(self.W.copy())
            self.historico_bias.append(self.b.copy() if multiclasse else float(self.b))

            # Métricas da época
            y_pred_tr = self.predict(X_train)
            self.historico_acc_treino.append(accuracy_score(y_train, y_pred_tr) * 100.0)
            if X_test is not None:
                y_pred_te = self.predict(X_test)
                self.historico_acc_teste.append(accuracy_score(y_test, y_pred_te) * 100.0)
            self.historico_perda.append(self._calcular_perda(X_train, y_train))

        return self

    def predict_score(self, X):
        """Retorna as somas ponderadas lineares z."""
        if self.num_classes > 2:
            return np.dot(X, self.W.T) + self.b
        return np.dot(X, self.W) + self.b

    def predict(self, X):
        """Retorna as classes previstas 0, 1, ... K-1."""
        z = self.predict_score(X)
        if self.num_classes > 2:
            return np.argmax(z, axis=1)

        if self.ativacao == "degrau":
            return np.where(z >= 0.0, 1, 0)
        elif self.ativacao == "sigmoide":
            prob = self._ativar(z)
            return np.where(prob >= 0.5, 1, 0)
        elif self.ativacao == "tanh":
            return np.where(z >= 0.0, 1, 0)
        return np.where(z >= 0.0, 1, 0)

    def _calcular_perda(self, X, y):
        """Calcula o erro médio quadrático para acompanhamento da curva de perda."""
        N = len(y)
        if self.num_classes > 2:
            perda = 0.0
            for k in range(self.num_classes):
                t_k = (y == k).astype(np.float64)
                z_k = np.dot(X, self.W[k]) + self.b[k]
                out_k = self._ativar(z_k)
                perda += np.mean((t_k - out_k) ** 2)
            return float(perda / self.num_classes)
        else:
            t = y.astype(np.float64)
            if self.ativacao == "tanh":
                t = np.where(y == 1, 1.0, -1.0)
            z = np.dot(X, self.W) + self.b
            out = self._ativar(z)
            return float(np.mean((t - out) ** 2))


# ==============================================================================
# 2. MLP EM PYTORCH
# ==============================================================================

class PyTorchMLP(nn.Module):
    """Rede Neural Multicamadas (MLP) com camadas ocultas configuráveis."""

    def __init__(self, in_features, hidden_dims, num_classes, ativacao_nome="relu"):
        super().__init__()
        self.ativacao_nome = ativacao_nome

        if ativacao_nome == "sigmoide":
            act_fn = nn.Sigmoid
        elif ativacao_nome == "tanh":
            act_fn = nn.Tanh
        elif ativacao_nome == "leaky_relu":
            act_fn = lambda: nn.LeakyReLU(negative_slope=0.01)
        else:  # relu como padrão
            act_fn = nn.ReLU

        camadas = []
        ant = in_features
        for h in hidden_dims:
            camadas.append(nn.Linear(ant, h))
            camadas.append(act_fn())
            ant = h

        self.ocultas = nn.Sequential(*camadas)
        self.saida = nn.Linear(ant, num_classes)

    def forward(self, x):
        h = self.ocultas(x)
        return self.saida(h)


# ==============================================================================
# 3. CNN EM PYTORCH
# ==============================================================================

class PyTorchCNN(nn.Module):
    """Rede Convolucional simples: 1 Conv2D + Ativação + MaxPool + Linear."""

    def __init__(self, in_channels=1, n_filtros=8, kernel_size=3, num_classes=10,
                 ativacao_nome="relu"):
        super().__init__()
        self.ativacao_nome = ativacao_nome
        pad = kernel_size // 2

        self.conv = nn.Conv2d(in_channels, n_filtros, kernel_size=kernel_size, padding=pad)

        if ativacao_nome == "sigmoide":
            self.act = nn.Sigmoid()
        elif ativacao_nome == "tanh":
            self.act = nn.Tanh()
        elif ativacao_nome == "leaky_relu":
            self.act = nn.LeakyReLU(negative_slope=0.01)
        else:
            self.act = nn.ReLU()

        # Entrada 8x8 com pooling 2x2 resulta em mapa 4x4
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(n_filtros * 4 * 4, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ==============================================================================
# FUNÇÕES DE EXECUÇÃO E COLETA DE HISTÓRICO
# ==============================================================================

def treinar_perceptron(X_train, y_train, X_test, y_test, ativacao="sigmoide",
                       taxa=0.1, epocas=50, lote=16, semente=42):
    """Treina o Perceptron e retorna todos os dados, métricas e histórico."""
    modelo = PerceptronNumPy(
        ativacao=ativacao,
        taxa_aprendizado=taxa,
        epocas=epocas,
        tamanho_lote=lote,
        semente=semente
    )
    modelo.fit(X_train, y_train, X_test, y_test)

    y_pred_teste = modelo.predict(X_test)
    y_pred_treino = modelo.predict(X_train)

    acc_treino = float(accuracy_score(y_train, y_pred_treino) * 100.0)
    acc_teste = float(accuracy_score(y_test, y_pred_teste) * 100.0)
    perda_final = float(modelo.historico_perda[-1])

    # Monta histórico de parâmetros do neurônio 0 (ou único)
    multiclasse = modelo.num_classes > 2
    epocas_lista = list(range(len(modelo.historico_pesos)))

    # Extrai trajetória do neurônio 0
    trajetoria_bias = []
    trajetoria_pesos = []
    for ep in epocas_lista:
        b_ep = modelo.historico_bias[ep]
        W_ep = modelo.historico_pesos[ep]
        if multiclasse:
            trajetoria_bias.append(float(b_ep[0]))
            trajetoria_pesos.append([float(w) for w in W_ep[0]])
        else:
            trajetoria_bias.append(float(b_ep))
            trajetoria_pesos.append([float(w) for w in W_ep])

    tabela = _montar_tabela_parametros(epocas_lista, trajetoria_bias, trajetoria_pesos)

    # Histórico de pesos completo para desenhar retas de fronteira
    historico_fronteira = []
    for ep in [0, 1, 5, epocas]:
        if ep < len(modelo.historico_pesos):
            W_ep = modelo.historico_pesos[ep]
            b_ep = modelo.historico_bias[ep]
            if multiclasse:
                historico_fronteira.append((ep, W_ep[0].copy(), float(b_ep[0])))
            else:
                historico_fronteira.append((ep, W_ep.copy(), float(b_ep)))

    return {
        "modelo": "Perceptron",
        "acuracia_treino": round(acc_treino, 1),
        "acuracia_teste": round(acc_teste, 1),
        "perda_final": round(perda_final, 4),
        "y_teste_real": y_test,
        "y_teste_pred": y_pred_teste,
        "epocas": epocas_lista,
        "perdas": modelo.historico_perda,
        "acuracias_treino": modelo.historico_acc_treino,
        "acuracias_teste": modelo.historico_acc_teste,
        "normas_variacao": modelo.normas_variacao,
        "tabela_parametros": tabela,
        "trajetoria_bias": trajetoria_bias,
        "trajetoria_pesos": trajetoria_pesos,
        "historico_fronteira": historico_fronteira,
    }


def treinar_mlp(X_train, y_train, X_test, y_test, ativacao="relu",
                camadas_ocultas="16", taxa=0.01, epocas=50, lote=16, semente=42):
    """Treina o MLP em PyTorch e retorna métricas, tabelas e histórico."""
    torch.manual_seed(semente)
    np.random.seed(semente)

    # Interpreta camadas ocultas, ex: "16" ou "16, 8"
    try:
        dims_ocultas = [int(x.strip()) for x in camadas_ocultas.split(",") if x.strip()]
        if not dims_ocultas:
            dims_ocultas = [16]
    except Exception:
        dims_ocultas = [16]

    in_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    rede = PyTorchMLP(in_dim, dims_ocultas, num_classes, ativacao_nome=ativacao)
    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(rede.parameters(), lr=taxa)

    # Tensores
    t_Xtr = torch.tensor(X_train, dtype=torch.float32)
    t_ytr = torch.tensor(y_train, dtype=torch.long)
    t_Xte = torch.tensor(X_test, dtype=torch.float32)
    t_yte = torch.tensor(y_test, dtype=torch.long)

    dataset = TensorDataset(t_Xtr, t_ytr)
    loader = DataLoader(dataset, batch_size=lote, shuffle=True)

    # Referência da primeira camada Linear
    primeira_linear = rede.ocultas[0]

    # Estado na Época 0
    w_inicial = primeira_linear.weight.detach().cpu().numpy().copy()
    b_inicial = primeira_linear.bias.detach().cpu().numpy().copy()

    trajetoria_bias = [float(b_inicial[0])]
    trajetoria_pesos = [[float(w) for w in w_inicial[0]]]

    perdas = []
    accs_tr = []
    accs_te = []
    normas = []

    # Avaliação Época 0
    rede.eval()
    with torch.no_grad():
        out_tr = rede(t_Xtr)
        loss_0 = criterio(out_tr, t_ytr).item()
        pred_tr = torch.argmax(out_tr, dim=1).numpy()
        pred_te = torch.argmax(rede(t_Xte), dim=1).numpy()
        perdas.append(loss_0)
        accs_tr.append(accuracy_score(y_train, pred_tr) * 100.0)
        accs_te.append(accuracy_score(y_test, pred_te) * 100.0)

    # Loop de treino
    for ep in range(1, epocas + 1):
        rede.train()
        w_anterior = primeira_linear.weight.detach().cpu().numpy().copy()

        for batch_x, batch_y in loader:
            otimizador.zero_grad()
            saida = rede(batch_x)
            loss = criterio(saida, batch_y)
            loss.backward()
            otimizador.step()

        # Variação na primeira camada
        w_atual = primeira_linear.weight.detach().cpu().numpy().copy()
        b_atual = primeira_linear.bias.detach().cpu().numpy().copy()
        norma_delta = float(np.linalg.norm(w_atual - w_anterior))
        normas.append(norma_delta)

        trajetoria_bias.append(float(b_atual[0]))
        trajetoria_pesos.append([float(w) for w in w_atual[0]])

        # Avaliação da época
        rede.eval()
        with torch.no_grad():
            out_tr = rede(t_Xtr)
            loss_ep = criterio(out_tr, t_ytr).item()
            pred_tr = torch.argmax(out_tr, dim=1).numpy()
            pred_te = torch.argmax(rede(t_Xte), dim=1).numpy()
            perdas.append(loss_ep)
            accs_tr.append(accuracy_score(y_train, pred_tr) * 100.0)
            accs_te.append(accuracy_score(y_test, pred_te) * 100.0)

    epocas_lista = list(range(len(perdas)))
    tabela = _montar_tabela_parametros(epocas_lista, trajetoria_bias, trajetoria_pesos)

    w_final = primeira_linear.weight.detach().cpu().numpy().copy()

    return {
        "modelo": "MLP",
        "acuracia_treino": round(accs_tr[-1], 1),
        "acuracia_teste": round(accs_te[-1], 1),
        "perda_final": round(perdas[-1], 4),
        "y_teste_real": y_test,
        "y_teste_pred": pred_te,
        "epocas": epocas_lista,
        "perdas": perdas,
        "acuracias_treino": accs_tr,
        "acuracias_teste": accs_te,
        "normas_variacao": normas,
        "tabela_parametros": tabela,
        "trajetoria_bias": trajetoria_bias,
        "trajetoria_pesos": trajetoria_pesos,
        "pesos_iniciais": w_inicial,
        "pesos_finais": w_final,
    }


def treinar_cnn(X_train, y_train, X_test, y_test, ativacao="relu",
                n_filtros=8, kernel_size=3, taxa=0.01, epocas=20, lote=32, semente=42):
    """Treina a CNN em PyTorch e retorna filtros antes/depois, métricas e histórico."""
    torch.manual_seed(semente)
    np.random.seed(semente)

    num_classes = len(np.unique(y_train))
    rede = PyTorchCNN(in_channels=1, n_filtros=n_filtros, kernel_size=kernel_size,
                      num_classes=num_classes, ativacao_nome=ativacao)
    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(rede.parameters(), lr=taxa)

    t_Xtr = torch.tensor(X_train, dtype=torch.float32)
    t_ytr = torch.tensor(y_train, dtype=torch.long)
    t_Xte = torch.tensor(X_test, dtype=torch.float32)
    t_yte = torch.tensor(y_test, dtype=torch.long)

    dataset = TensorDataset(t_Xtr, t_ytr)
    loader = DataLoader(dataset, batch_size=lote, shuffle=True)

    # Filtros na época 0
    filtros_iniciais = rede.conv.weight.detach().cpu().numpy().copy()
    b_inicial = rede.conv.bias.detach().cpu().numpy().copy()

    # Rastreia parâmetros do Filtro 0 (kernel achatado)
    trajetoria_bias = [float(b_inicial[0])]
    trajetoria_pesos = [[float(w) for w in filtros_iniciais[0, 0].flatten()]]

    perdas = []
    accs_tr = []
    accs_te = []
    normas = []

    # Avaliação na Época 0
    rede.eval()
    with torch.no_grad():
        out_tr = rede(t_Xtr)
        loss_0 = criterio(out_tr, t_ytr).item()
        pred_tr = torch.argmax(out_tr, dim=1).numpy()
        pred_te = torch.argmax(rede(t_Xte), dim=1).numpy()
        perdas.append(loss_0)
        accs_tr.append(accuracy_score(y_train, pred_tr) * 100.0)
        accs_te.append(accuracy_score(y_test, pred_te) * 100.0)

    # Loop de treino
    for ep in range(1, epocas + 1):
        rede.train()
        w_anterior = rede.conv.weight.detach().cpu().numpy().copy()

        for batch_x, batch_y in loader:
            otimizador.zero_grad()
            saida = rede(batch_x)
            loss = criterio(saida, batch_y)
            loss.backward()
            otimizador.step()

        w_atual = rede.conv.weight.detach().cpu().numpy().copy()
        b_atual = rede.conv.bias.detach().cpu().numpy().copy()

        norma_delta = float(np.linalg.norm(w_atual - w_anterior))
        normas.append(norma_delta)

        trajetoria_bias.append(float(b_atual[0]))
        trajetoria_pesos.append([float(w) for w in w_atual[0, 0].flatten()])

        rede.eval()
        with torch.no_grad():
            out_tr = rede(t_Xtr)
            loss_ep = criterio(out_tr, t_ytr).item()
            pred_tr = torch.argmax(out_tr, dim=1).numpy()
            pred_te = torch.argmax(rede(t_Xte), dim=1).numpy()
            perdas.append(loss_ep)
            accs_tr.append(accuracy_score(y_train, pred_tr) * 100.0)
            accs_te.append(accuracy_score(y_test, pred_te) * 100.0)

    epocas_lista = list(range(len(perdas)))
    tabela = _montar_tabela_parametros(epocas_lista, trajetoria_bias, trajetoria_pesos)
    filtros_finais = rede.conv.weight.detach().cpu().numpy().copy()

    return {
        "modelo": "CNN",
        "acuracia_treino": round(accs_tr[-1], 1),
        "acuracia_teste": round(accs_te[-1], 1),
        "perda_final": round(perdas[-1], 4),
        "y_teste_real": y_test,
        "y_teste_pred": pred_te,
        "epocas": epocas_lista,
        "perdas": perdas,
        "acuracias_treino": accs_tr,
        "acuracias_teste": accs_te,
        "normas_variacao": normas,
        "tabela_parametros": tabela,
        "trajetoria_bias": trajetoria_bias,
        "trajetoria_pesos": trajetoria_pesos,
        "filtros_iniciais": filtros_iniciais,
        "filtros_finais": filtros_finais,
    }


def _montar_tabela_parametros(epocas_lista, trajetoria_bias, trajetoria_pesos):
    """Monta a tabela com épocas selecionadas (ex: 0, 1, 2, 5, 25, final)

    e a linha com o Delta total, exatamente como no exemplo do enunciado.
    """
    total_epocas = len(epocas_lista) - 1
    # Pontos de amostragem padrão
    candidatos = [0, 1, 2, 5, 10, 25, 50, 100, 150, 200]
    selecionadas = [ep for ep in candidatos if ep < total_epocas]
    if total_epocas not in selecionadas:
        selecionadas.append(total_epocas)

    num_pesos = min(4, len(trajetoria_pesos[0]))
    cabecalho = ["Época", "bias"] + [f"w{i+1}" for i in range(num_pesos)]

    linhas = []
    for ep in selecionadas:
        idx = epocas_lista.index(ep)
        bias_val = f"{trajetoria_bias[idx]:+.3f}"
        pesos_vals = [f"{trajetoria_pesos[idx][i]:+.3f}" for i in range(num_pesos)]
        linhas.append({
            "epoca": str(ep),
            "bias": bias_val,
            "pesos": pesos_vals
        })

    # Linha Delta total: valor_final - valor_inicial
    delta_bias = trajetoria_bias[-1] - trajetoria_bias[0]
    deltas_pesos = [trajetoria_pesos[-1][i] - trajetoria_pesos[0][i] for i in range(num_pesos)]

    delta_linha = {
        "epoca": "Δ total",
        "bias": f"{delta_bias:+.3f}",
        "pesos": [f"{dp:+.3f}" for dp in deltas_pesos]
    }

    return {
        "cabecalho": cabecalho,
        "linhas": linhas,
        "delta": delta_linha
    }
