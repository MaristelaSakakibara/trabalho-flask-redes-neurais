# Trabalho Integrador: Aplicação Flask para Redes Neurais
**Disciplina:** Aprendizado de Máquina (PEPAPRM)  
**Tema:** Perceptron, MLP e CNN  

---

## 📌 1. Visão Geral
Aplicação web interativa desenvolvida em **Flask** que permite treinar, avaliar e analisar três arquiteturas fundamentais de redes neurais:
1. **Perceptron de Camada Única** (implementado em **NumPy puro**, sem bibliotecas de alto nível);
2. **Perceptron Multicamadas (MLP)** com camadas ocultas configuráveis;
3. **Rede Neural Convolucional (CNN)** com 1 camada convolucional e visualização dos filtros aprendidos.

O foco central da ferramenta é demonstrar **o que acontece dentro da rede** ao longo do treinamento, exibindo métricas, gráficos de convergência, trajetórias e tabelas de evolução dos pesos e bias, norma da variação dos pesos e matrizes de confusão.

---

## 🚀 2. Como Instalar e Executar

### Pré-requisitos
- Python 3.10 ou superior
- Pip e Venv instalados

### Passo a Passo

1. **Crie e ative um ambiente virtual:**
   ```bash
   # No Linux / macOS:
   python3 -m venv .venv
   source .venv/bin/activate

   # No Windows:
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Inicie o servidor Flask:**
   ```bash
   python app.py
   ```

4. **Acesse no seu navegador:**
   Abra [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📂 3. Estrutura do Projeto

```text
Arquivos-20261109/
├── app.py              # Rotas da aplicação Flask (GET para interface e POST para treinamento)
├── modelos.py          # Perceptron (NumPy), MLP (PyTorch) e CNN (PyTorch) com histórico época a época
├── dados.py            # Carregamento e pré-processamento das bases de dados (RF2 e RF5)
├── graficos.py         # Geração de gráficos com Matplotlib convertidos para Base64 (RF7, RF8, RF9, RF10, RF11)
├── templates/
│   └── index.html      # Interface web responsiva em Bootstrap 5 com formulário e painel de resultados
├── requirements.txt    # Lista de dependências do projeto
├── README.md           # Instruções completas de uso e documentação
├── alunos.csv          # Base de dados tabular local (classificação binária de aprovação)
├── grupo.csv           # Base de dados tabular local (classificação binária de clientes)
└── aluguel.csv         # Base de dados de exemplo
```

---

## ✅ 4. Requisitos Atendidos

| Código | Requisito | Como foi implementado |
|---|---|---|
| **RF1** | Escolha do modelo | Seleção entre Perceptron, MLP e CNN na interface |
| **RF2** | Escolha da base de dados | Suporte a Iris, Câncer de Mama, Dígitos 8x8, Alunos, Grupo e Upload de CSV exclusivo |
| **RF3** | Função de ativação | Degrau, Sigmoide, Tanh, ReLU e Leaky ReLU, com validação inteligente na interface |
| **RF4** | Hiperparâmetros configuráveis | Taxa de aprendizado, épocas, tamanho do lote, split treino/teste, neurônios ocultos e filtros |
| **RF5** | Pré-processamento sem data leakage | `StandardScaler` ajustado **apenas no treino** para dados tabulares; imagens em `[0, 1]` |
| **RF6** | Treino e exibição em página única | Requisição POST com retorno dinâmico na mesma página |
| **RF7** | Métricas de avaliação | Acurácia de treino, acurácia de teste, perda final e matriz de confusão gráfica |
| **RF8** | Gráficos de perda e acurácia | Subplots lado a lado com curva de perda e curva de acurácia por época |
| **RF9** | Ajuste dos parâmetros | Tabela com bias e pesos nas épocas 0, 1, 2, 5... mais linha de Δ total; gráfico de trajetória e norma \|\|ΔW\|\| |
| **RF10** | Fronteira de decisão 2D | Gráfico com dispersão dos pontos de treino e retas em épocas 0, 1, 5 e final para 2 atributos |
| **RF11** | Visualização de filtros CNN | Painel com imagens dos kernels antes do treino (Época 0) e depois do treino |
| **RF12** | Tratamento de erros | Alertas amigáveis no topo da página para entradas inválidas sem derrubar a aplicação |

---

## 📝 5. Subsídios para o Relatório e Apresentação (Seção 7 do Enunciado)

### 7.1 Cálculo Manual de uma Atualização no Perceptron
- **Exemplo com amostra:** Seja uma amostra normalizada $x = (x_1, x_2, \dots, x_d)$ e classe $t \in \{0, 1\}$.
- **Saída do neurônio:** $z = b + \sum_{i} w_i x_i$. Aplicando a função degrau:
  $$y = 1 \text{ se } z \ge 0 \text{ senão } 0$$
- **Erro:** $e = t - y$.
- **Variação dos pesos e bias:**
  $$\Delta w_i = \eta \cdot e \cdot x_i$$
  $$\Delta b = \eta \cdot e$$
- **Novos valores:** $w_i^{(novo)} = w_i + \Delta w_i$, $b^{(novo)} = b + \Delta b$.

### 7.2 Análise Comparativa
1. **Perceptron (Degrau vs Sigmoide vs Tanh):**
   - No degrau, as correções são discretas e ocorrem apenas em caso de erro de classificação ($t - y \neq 0$).
   - Na sigmoide e tanh, a atualização usa a regra delta ponderada pela derivada da ativação, permitindo convergência suave e contínua das fronteiras de decisão.
2. **Perceptron vs MLP:**
   - O Perceptron encontra apenas separadores lineares (retas/hiperplanos).
   - O MLP combina múltiplos neurônios com ativações não-lineares, conseguindo aprender fronteiras curvas e resolver problemas não linearmente separáveis (como o XOR).
3. **Ativações no MLP (Sigmoide vs Tanh vs ReLU):**
   - ReLU evita o problema de desaparecimento do gradiente (*vanishing gradient*), pois sua derivada é constante ($1.0$) para valores positivos, acelerando significativamente a convergência.
4. **Variação da Taxa de Aprendizado ($\eta$):**
   - Valores muito baixos (ex: $0.001$) tornam o aprendizado excessivamente lento;
   - Valores ideais (ex: $0.05$ a $0.1$) convergem suavemente;
   - Valores muito altos (ex: $1.0$ ou mais) causam oscilações bruscas e divergência da perda.
5. **O que os filtros da CNN aprenderam:**
   - Na Época 0, os filtros possuem pesos aleatórios sem padrão definido (ruído).
   - Ao final do treino, os filtros especializam-se como detectores de características locais (bordas verticais, horizontais e contrastes de intensidade).
