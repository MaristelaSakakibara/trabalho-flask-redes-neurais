"""graficos.py - Geração de gráficos com Matplotlib e conversão para Base64.

Funções para:
- Curvas de perda e acurácia por época (RF8)
- Trajetória dos pesos e do bias ao longo das épocas (RF9)
- Norma da variação dos pesos por época (RF9)
- Matriz de confusão no teste (RF7)
- Fronteira de decisão do Perceptron em 2D com retas de épocas (RF10)
- Visualização dos filtros da CNN antes e depois do treino (RF11)
- Mapa de calor dos pesos para MLP (Critério de avaliação nível 2.0)
"""

import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend não interativo para servidores web
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def _fig_para_base64(fig):
    """Converte uma figura do Matplotlib para string Base64 em formato PNG data-uri."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
    buffer.seek(0)
    dados_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{dados_b64}"


def grafico_perda_acuracia(epocas, perdas, acc_treino, acc_teste):
    """Gera gráfico com 2 subplots: Curva de Perda e Curva de Acurácia."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Gráfico de Perda
    ax1.plot(epocas, perdas, color="#d9534f", linewidth=2, label="Perda (Loss)")
    ax1.set_title("Evolução da Perda por Época", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Perda")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Gráfico de Acurácia
    ax2.plot(epocas, acc_treino, color="#0275d8", linewidth=2, label="Treino")
    if acc_teste:
        ax2.plot(epocas, acc_teste, color="#5cb85c", linewidth=2, linestyle="--", label="Teste")
    ax2.set_title("Evolução da Acurácia (%) por Época", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Acurácia (%)")
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    return _fig_para_base64(fig)


def grafico_evolucao_parametros(epocas, trajetoria_bias, trajetoria_pesos):
    """Gera o gráfico de linhas com a trajetória dos pesos e do bias."""
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(epocas, trajetoria_bias, label="bias", linewidth=2.2, color="#000000")

    cores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    num_pesos = min(4, len(trajetoria_pesos[0]))
    for i in range(num_pesos):
        vals = [trajetoria_pesos[ep][i] for ep in range(len(epocas))]
        cor = cores[i % len(cores)]
        ax.plot(epocas, vals, label=f"w{i+1}", linewidth=1.8, color=cor)

    ax.set_title("Evolução dos Pesos e do Bias ao Longo das Épocas", fontsize=11, fontweight="bold")
    ax.set_xlabel("Época")
    ax.set_ylabel("Valor do Parâmetro")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    plt.tight_layout()
    return _fig_para_base64(fig)


def grafico_norma_variacao(normas_variacao):
    """Gera o gráfico da norma da variação dos pesos por época (||ΔW||)."""
    fig, ax = plt.subplots(figsize=(7, 3.5))

    epocas = list(range(1, len(normas_variacao) + 1))
    ax.plot(epocas, normas_variacao, color="#8a2be2", marker="o", markersize=3, linewidth=1.8)
    ax.set_title("Norma da Variação dos Pesos (||ΔW||) por Época", fontsize=11, fontweight="bold")
    ax.set_xlabel("Época")
    ax.set_ylabel("||W_t - W_{t-1}||")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    return _fig_para_base64(fig)


def grafico_matriz_confusao(y_real, y_pred, nomes_classes):
    """Gera o heatmap da matriz de confusão com rótulos e contagens."""
    cm = confusion_matrix(y_real, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))

    cax = ax.matshow(cm, cmap="Blues", alpha=0.8)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    # Rótulos dos eixos
    ax.set_xticks(range(len(nomes_classes)))
    ax.set_yticks(range(len(nomes_classes)))
    ax.set_xticklabels(nomes_classes, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(nomes_classes, fontsize=9)
    ax.set_xlabel("Previsto", fontweight="bold", labelpad=8)
    ax.set_ylabel("Real", fontweight="bold")
    ax.set_title("Matriz de Confusão (Conjunto de Teste)", fontsize=11, fontweight="bold", pad=20)

    # Anotações nas células
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cor_texto = "white" if cm[i, j] > cm.max() / 2.0 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=cor_texto, fontsize=10, fontweight="bold")

    plt.tight_layout()
    return _fig_para_base64(fig)


def grafico_fronteira_decisao(X_train, y_train, historico_fronteira, idx1=0, idx2=1, nomes_features=None):
    """Desenha os pontos de treino 2D e as retas da fronteira em épocas selecionadas (ex: 0, 1, 5, final)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    x_min, x_max = X_train[:, idx1].min() - 0.5, X_train[:, idx1].max() + 0.5
    y_min, y_max = X_train[:, idx2].min() - 0.5, X_train[:, idx2].max() + 0.5

    # Plota os dados de treino
    classes_unicas = np.unique(y_train)
    cores_pts = ["#d9534f", "#0275d8", "#5cb85c", "#f0ad4e"]
    for i, c in enumerate(classes_unicas):
        mask = (y_train == c)
        cor = cores_pts[i % len(cores_pts)]
        ax.scatter(X_train[mask, idx1], X_train[mask, idx2], color=cor,
                   edgecolor="k", alpha=0.75, label=f"Classe {c}")

    # Retas da fronteira nas épocas selecionadas
    # Equação: w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b) / w2
    x_vals = np.linspace(x_min, x_max, 100)
    estilos = [("Época 0", "--", "#999999"), ("Época 1", ":", "#f0ad4e"),
               ("Época 5", "-.", "#0275d8"), ("Época Final", "-", "#d9534f")]

    for ep_tupla, (nome_ep, estilo, cor_linha) in zip(historico_fronteira, estilos):
        ep, W_ep, b_ep = ep_tupla
        w1 = W_ep[idx1]
        w2 = W_ep[idx2]
        if abs(w2) > 1e-5:
            y_vals = -(w1 * x_vals + b_ep) / w2
            # Filtra valores dentro dos limites visíveis
            ax.plot(x_vals, y_vals, linestyle=estilo, color=cor_linha,
                    linewidth=2, label=f"Reta (Época {ep})")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    nome_x = nomes_features[idx1] if nomes_features and idx1 < len(nomes_features) else f"Atributo {idx1+1}"
    nome_y = nomes_features[idx2] if nomes_features and idx2 < len(nomes_features) else f"Atributo {idx2+1}"
    ax.set_xlabel(nome_x)
    ax.set_ylabel(nome_y)
    ax.set_title("Fronteira de Decisão do Perceptron em 2D", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    return _fig_para_base64(fig)


def grafico_filtros_cnn(filtros_iniciais, filtros_finais):
    """Mostra os filtros convolucionais antes e depois do treinamento como imagens."""
    n_filtros = filtros_iniciais.shape[0]
    n_mostrar = min(8, n_filtros)

    fig, axes = plt.subplots(2, n_mostrar, figsize=(n_mostrar * 1.5, 3.5))

    for i in range(n_mostrar):
        # Filtro inicial (linha 0)
        ax_ini = axes[0, i] if n_mostrar > 1 else axes[0]
        f_ini = filtros_iniciais[i, 0]
        ax_ini.imshow(f_ini, cmap="viridis", interpolation="nearest")
        ax_ini.set_title(f"F{i+1} Ini", fontsize=9)
        ax_ini.axis("off")

        # Filtro final (linha 1)
        ax_fim = axes[1, i] if n_mostrar > 1 else axes[1]
        f_fim = filtros_finais[i, 0]
        ax_fim.imshow(f_fim, cmap="viridis", interpolation="nearest")
        ax_fim.set_title(f"F{i+1} Fim", fontsize=9)
        ax_fim.axis("off")

    fig.suptitle("Filtros da Camada Convolucional (Linha 1: Antes | Linha 2: Depois)",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    return _fig_para_base64(fig)


def grafico_heatmap_pesos(pesos_iniciais, pesos_finais):
    """Mostra heatmap dos pesos da primeira camada oculta do MLP antes e depois."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    # Matriz inicial
    im1 = ax1.imshow(pesos_iniciais, cmap="coolwarm", aspect="auto")
    ax1.set_title("Pesos Iniciais (Época 0)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Entradas")
    ax1.set_ylabel("Neurônios Ocultos")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Matriz final
    im2 = ax2.imshow(pesos_finais, cmap="coolwarm", aspect="auto")
    ax2.set_title("Pesos Finais", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Entradas")
    ax2.set_ylabel("Neurônios Ocultos")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle("Mapa de Calor dos Pesos da Camada Oculta (MLP)", fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    return _fig_para_base64(fig)
