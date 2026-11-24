"""dados.py - Carregamento e pré-processamento das bases de dados.

Contempla:
- Bases embutidas: Iris, Câncer de Mama e Dígitos 8x8.
- Bases em CSV locais: alunos.csv, grupo.csv.
- Base CSV exclusiva do aluno enviada por upload.
- Pré-processamento sem vazamento de dados (StandardScaler ajustado apenas no treino).
- Imagens escaladas para [0, 1].
"""

import io
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))


def carregar_base(nome_base, arquivo_upload=None):
    """Carrega a base de dados selecionada e retorna X, y, nomes dos atributos,

    nomes das classes e o tipo da base ('tabular' ou 'imagem').
    """
    if nome_base == "iris":
        dados = load_iris()
        X = dados.data
        y = dados.target
        features = [str(f) for f in dados.feature_names]
        classes = [str(c) for c in dados.target_names]
        tipo = "tabular"

    elif nome_base == "cancer":
        dados = load_breast_cancer()
        X = dados.data
        y = dados.target
        features = [str(f) for f in dados.feature_names]
        classes = [str(c) for c in dados.target_names]
        tipo = "tabular"

    elif nome_base == "digits":
        dados = load_digits()
        # Imagens 8x8 originais
        X = dados.images.astype(np.float32)
        y = dados.target
        features = [f"pixel_{i}" for i in range(64)]
        classes = [str(i) for i in range(10)]
        tipo = "imagem"

    elif nome_base == "alunos":
        caminho = os.path.join(DIRETORIO_ATUAL, "alunos.csv")
        if not os.path.exists(caminho):
            raise FileNotFoundError("Arquivo alunos.csv não encontrado na pasta.")
        df = pd.read_csv(caminho)
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(int)
        features = list(df.columns[:-1])
        classes = ["Reprovado (0)", "Aprovado (1)"]
        tipo = "tabular"

    elif nome_base == "grupo":
        caminho = os.path.join(DIRETORIO_ATUAL, "grupo.csv")
        if not os.path.exists(caminho):
            raise FileNotFoundError("Arquivo grupo.csv não encontrado na pasta.")
        df = pd.read_csv(caminho)
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(int)
        features = list(df.columns[:-1])
        classes = ["Padrão (0)", "Premium (1)"]
        tipo = "tabular"

    elif nome_base == "upload":
        if arquivo_upload is None or arquivo_upload.filename == "":
            raise ValueError("Nenhum arquivo CSV foi enviado para a base exclusiva.")

        # Leitura do CSV enviado
        conteudo = arquivo_upload.read().decode("utf-8-sig")
        primeira_linha = conteudo.splitlines()[0] if conteudo else ""
        sep = ";" if primeira_linha.count(";") > primeira_linha.count(",") else ","

        df = pd.read_csv(io.StringIO(conteudo), sep=sep)
        if df.shape[1] < 2:
            raise ValueError("O CSV deve conter pelo menos uma coluna de atributo e uma coluna de classe.")

        # Substitui vírgula decimal por ponto nas colunas numéricas
        for col in df.columns[:-1]:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

        X = df.iloc[:, :-1].values.astype(np.float32)
        y_raw = df.iloc[:, -1].values

        # Se a classe for texto, codifica para 0, 1, 2...
        if y_raw.dtype == object or isinstance(y_raw[0], str):
            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            classes = [str(c) for c in le.classes_]
        else:
            classes_unicas = np.unique(y_raw)
            # Mapeia para 0, 1, ... sequencial
            mapeamento = {val: idx for idx, val in enumerate(sorted(classes_unicas))}
            y = np.array([mapeamento[val] for val in y_raw], dtype=int)
            classes = [str(c) for c in sorted(classes_unicas)]

        features = list(df.columns[:-1])
        tipo = "tabular"

    else:
        raise ValueError(f"Base de dados desconhecida: {nome_base}")

    return X, y, features, classes, tipo


def preparar_dados(X, y, tipo, prop_teste=0.2, semente=42, modelo="perceptron"):
    """Divide os dados em treino e teste e aplica o pré-processamento adequado.

    - Tabular: padronização com média e desvio calculados APENAS no treino (RF5).
    - Imagem: normalização para o intervalo [0, 1] e formatação para CNN ou MLP.
    """
    # Divisão treino/teste
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=prop_teste, random_state=semente, stratify=y
        )
    except Exception:
        # Se a estratificação falhar (ex: classes com apenas 1 amostra no CSV)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=prop_teste, random_state=semente
        )

    scaler = None
    if tipo == "tabular":
        # Padronização z = (x - média) / desvio calculada apenas no treino
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    elif tipo == "imagem":
        # Imagens de load_digits possuem escala 0 a 16 -> normaliza para [0, 1]
        X_train = X_train / 16.0
        X_test = X_test / 16.0

        if modelo == "cnn":
            # Formato convolucional: (N, canais=1, altura=8, largura=8)
            X_train = X_train.reshape(-1, 1, 8, 8).astype(np.float32)
            X_test = X_test.reshape(-1, 1, 8, 8).astype(np.float32)
        else:
            # Perceptron ou MLP recebem os pixels achatados (64)
            X_train = X_train.reshape(-1, 64).astype(np.float32)
            X_test = X_test.reshape(-1, 64).astype(np.float32)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }
