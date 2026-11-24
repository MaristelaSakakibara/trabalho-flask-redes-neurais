"""app.py - Aplicação Flask para treinamento e análise de Perceptron, MLP e CNN.

Execução:
    python app.py
Acesse:
    http://127.0.0.1:5000
"""

import os
from flask import Flask, render_template, request
import dados
import modelos
import graficos

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Limite de 16MB para upload


@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    erro = None

    # Parâmetros padrão para preencher o formulário
    valores = {
        "modelo": "perceptron",
        "base": "iris",
        "ativacao": "sigmoide",
        "taxa": "0.1",
        "epocas": "50",
        "lote": "16",
        "prop_teste": "0.25",
        "semente": "42",
        "camadas_ocultas": "16",
        "n_filtros": "8",
        "kernel_size": "3",
        "feat_x": "0",
        "feat_y": "1",
    }

    if request.method == "POST":
        try:
            # Captura valores informados pelo usuário
            modelo_escolhido = request.form.get("modelo", "perceptron")
            base_escolhida = request.form.get("base", "iris")
            ativacao = request.form.get("ativacao", "sigmoide")
            taxa = float(request.form.get("taxa", 0.1))
            epocas = int(request.form.get("epocas", 50))
            lote = int(request.form.get("lote", 16))
            prop_teste = float(request.form.get("prop_teste", 0.25))
            semente = int(request.form.get("semente", 42))

            # Atualiza dicionário de formulário para manter os valores na tela
            valores.update({
                "modelo": modelo_escolhido,
                "base": base_escolhida,
                "ativacao": ativacao,
                "taxa": str(taxa),
                "epocas": str(epocas),
                "lote": str(lote),
                "prop_teste": str(prop_teste),
                "semente": str(semente),
                "camadas_ocultas": request.form.get("camadas_ocultas", "16"),
                "n_filtros": request.form.get("n_filtros", "8"),
                "kernel_size": request.form.get("kernel_size", "3"),
                "feat_x": request.form.get("feat_x", "0"),
                "feat_y": request.form.get("feat_y", "1"),
            })

            # -------------------------------------------------------------
            # RF12: Tratamento e validação de regras de negócio
            # -------------------------------------------------------------
            if modelo_escolhido == "cnn" and base_escolhida != "digits":
                raise ValueError(
                    "O modelo CNN requer uma base de imagens (como Dígitos 8x8). "
                    "Bases tabulares não possuem estrutura bidimensional de pixels para convolução. "
                    "Para bases tabulares, utilize o Perceptron ou o MLP."
                )

            if modelo_escolhido in ["mlp", "cnn"] and ativacao == "degrau":
                raise ValueError(
                    "A função de ativação degrau não pode ser usada em redes MLP ou CNN, pois sua "
                    "derivada é nula em quase toda parte (e indefinida em 0), impedindo o aprendizado "
                    "por descida do gradiente (backpropagation). Escolha Sigmoide, Tanh, ReLU ou Leaky ReLU."
                )

            # Limite preventivo de épocas para tempo web (Seção 2 do enunciado)
            if modelo_escolhido == "cnn" and epocas > 30:
                epocas = 30
                valores["epocas"] = "30"
            elif modelo_escolhido in ["perceptron", "mlp"] and epocas > 200:
                epocas = 200
                valores["epocas"] = "200"

            # -------------------------------------------------------------
            # Carregamento e Pré-processamento dos dados (RF2 e RF5)
            # -------------------------------------------------------------
            arquivo_upload = request.files.get("arquivo_upload") if base_escolhida == "upload" else None
            X, y, nomes_features, nomes_classes, tipo_base = dados.carregar_base(base_escolhida, arquivo_upload)

            dados_prep = dados.preparar_dados(
                X, y, tipo_base, prop_teste=prop_teste, semente=semente, modelo=modelo_escolhido
            )
            X_tr, y_tr = dados_prep["X_train"], dados_prep["y_train"]
            X_te, y_te = dados_prep["X_test"], dados_prep["y_test"]

            # -------------------------------------------------------------
            # Treinamento do Modelo Escolhido (RF1, RF3, RF4)
            # -------------------------------------------------------------
            grafico_extra = None
            titulo_grafico_extra = ""

            if modelo_escolhido == "perceptron":
                res = modelos.treinar_perceptron(
                    X_tr, y_tr, X_te, y_te,
                    ativacao=ativacao, taxa=taxa, epocas=epocas, lote=lote, semente=semente
                )

                # RF10: Fronteira de decisão do Perceptron com 2 atributos
                idx_x = int(valores["feat_x"]) if int(valores["feat_x"]) < X_tr.shape[1] else 0
                idx_y = int(valores["feat_y"]) if int(valores["feat_y"]) < X_tr.shape[1] else min(1, X_tr.shape[1] - 1)
                grafico_extra = graficos.grafico_fronteira_decisao(
                    X_tr, y_tr, res["historico_fronteira"],
                    idx1=idx_x, idx2=idx_y, nomes_features=nomes_features
                )
                titulo_grafico_extra = f"Fronteira de Decisão ({nomes_features[idx_x]} vs {nomes_features[idx_y]})"

            elif modelo_escolhido == "mlp":
                camadas_str = valores["camadas_ocultas"]
                res = modelos.treinar_mlp(
                    X_tr, y_tr, X_te, y_te,
                    ativacao=ativacao, camadas_ocultas=camadas_str,
                    taxa=taxa, epocas=epocas, lote=lote, semente=semente
                )
                # Mapa de calor dos pesos inicial x final
                grafico_extra = graficos.grafico_heatmap_pesos(res["pesos_iniciais"], res["pesos_finais"])
                titulo_grafico_extra = "Mapa de Calor dos Pesos da Camada Oculta (Inicial x Final)"

            elif modelo_escolhido == "cnn":
                n_filtros = int(valores["n_filtros"])
                k_size = int(valores["kernel_size"])
                res = modelos.treinar_cnn(
                    X_tr, y_tr, X_te, y_te,
                    ativacao=ativacao, n_filtros=n_filtros, kernel_size=k_size,
                    taxa=taxa, epocas=epocas, lote=lote, semente=semente
                )
                # RF11: Filtros da CNN antes e depois como imagens
                grafico_extra = graficos.grafico_filtros_cnn(res["filtros_iniciais"], res["filtros_finais"])
                titulo_grafico_extra = "Visualização dos Filtros Convolucionais (Antes vs Depois)"

            # -------------------------------------------------------------
            # Geração dos Gráficos (RF7, RF8, RF9)
            # -------------------------------------------------------------
            grafico_perda_acc = graficos.grafico_perda_acuracia(
                res["epocas"], res["perdas"], res["acuracias_treino"], res["acuracias_teste"]
            )
            grafico_params = graficos.grafico_evolucao_parametros(
                res["epocas"], res["trajetoria_bias"], res["trajetoria_pesos"]
            )
            grafico_norma = graficos.grafico_norma_variacao(res["normas_variacao"])
            grafico_cm = graficos.grafico_matriz_confusao(
                res["y_teste_real"], res["y_teste_pred"], nomes_classes
            )

            # Empacota resultados para renderização
            resultado = {
                "modelo": res["modelo"],
                "base": base_escolhida.capitalize(),
                "ativacao": ativacao,
                "acuracia_treino": res["acuracia_treino"],
                "acuracia_teste": res["acuracia_teste"],
                "perda_final": res["perda_final"],
                "tabela": res["tabela_parametros"],
                "grafico_perda_acc": grafico_perda_acc,
                "grafico_params": grafico_params,
                "grafico_norma": grafico_norma,
                "grafico_cm": grafico_cm,
                "grafico_extra": grafico_extra,
                "titulo_grafico_extra": titulo_grafico_extra,
            }

        except Exception as e:
            # RF12: Tratamento amigável de erros
            erro = str(e)

    return render_template("index.html", resultado=resultado, erro=erro, valores=valores)


if __name__ == "__main__":
    # Inicia o servidor Flask em modo debug
    app.run(debug=True, port=5000)
