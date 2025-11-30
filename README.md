Análise e Modelagem de Performance de Estudantes

Este projeto cobre o processamento, limpeza, análise exploratória e modelagem de um conjunto de dados sobre a performance de estudantes, culminando na implementação de dois algoritmos de regressão (KNN e Regressão Linear) do zero.

1. Preparação e Limpeza de Dados

O arquivo Student_Performance_sujo.csv foi processado para garantir a qualidade dos dados.

1.1 Limpeza e Feature Engineering
As seguintes operações foram realizadas no dataset:
- Carregamento e Limpeza Inicial: O arquivo foi carregado e os nomes das colunas foram padronizados.
- Remoção de Duplicatas: Foram eliminados os registros duplicados.
- Remoção de Outliers: Registros com valores em Sample_Question_Papers_Practiced acima do percentil 80 foram removidos.
- Tratamento de Faltantes: Valores NaN nas colunas numéricas foram preenchidos com a média da respectiva coluna.
- Conversão Categórica: A coluna Extracurricular_Activities foi binarizada: Yes $\to$ 1 e No $\to$ 0.
- Criação de Feature: Foi criada a coluna practice_per_hour (razão entre provas práticas e horas estudadas).
- Normalização Z-score: Todas as colunas de entrada (exceto Performance_Index) foram normalizadas usando Z-score ($z = \frac{x - \mu}{\sigma}$).
- Shape Final do Dataset Limpo: (9904, 7)

1.2 Gráfico de Funções de Ativação
Um gráfico comparativo foi gerado para visualizar as funções de ativação mais comuns em Redes Neurais, usando 1000 pontos no intervalo $[-5, 5]$.
- Funções Incluídas: Sigmoide, ReLU, Tangente Hiperbólica (Tanh), GeLU, Leaky ReLU e Swish.

2. Visualização de Dados (EDA)
   
Após a limpeza, foram gerados gráficos para entender a distribuição e a correlação dos dados.

2.1. Distribuição da Variável Alvo
Um Histograma foi criado para a variável alvo Performance_Index após a aplicação da transformação logarítmica ($\text{log}(1+x)$), visando uma distribuição mais próxima do ideal para modelos de Machine Learning.

2.2. Correlação entre Variáveis
Um Mapa de Calor (Heatmap) foi gerado para visualizar as correlações entre todas as variáveis numéricas do dataset limpo.

- Análise Chave: A coluna Performance_Index demonstrou forte correlação com as features Previous_Scores e Hours_Studied.

2.3. Boxplots de Dispersão
Boxplots foram gerados para cada variável numérica para visualizar a dispersão dos dados e a presença de outliers após a normalização Z-score.

3. Modelagem de Dados
  
Dois modelos de regressão foram implementados do zero em classes Python, utilizando apenas as bibliotecas básicas (numpy, pandas, matplotlib.pyplot):

3.1. Regressor KNN (KNNRegressor)
A classe implementa a lógica do KNN para regressão, incluindo:
- Método fit(X, y): Armazena os dados de treino.
- Método predict(X): Calcula a distância de cada ponto de teste para todos os pontos de treino, seleciona os $k$ vizinhos e retorna a média dos rótulos desses vizinhos.
- Métricas de Distância:Euclidiana: $d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$Dissimilaridade do Cosseno: $d(\mathbf{x}, \mathbf{y}) = 1 - \frac{\mathbf{x} \cdot \mathbf{y}}{||\mathbf{x}|| \cdot ||\mathbf{y}||}$

3.2. Regressão Linear pela Equação Normal (NormalLinearRegression)
A classe implementa a Regressão Linear sem o uso de gradiente descendente, mas sim pela solução analítica da Equação Normal.
- Método fit(X, y): Calcula os pesos $\mathbf{w}$ (incluindo o intercept) usando a fórmula:$$  \mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$
- Implementação: Utiliza-se np.linalg.pinv() (pseudoinversa) e uma coluna de $1$s é adicionada a $\mathbf{X}$ para calcular o intercept.

3.3. Avaliação dos Modelos
Os dados limpos foram divididos em 80% treino / 20% teste. Os seguintes modelos foram treinados e testados:
- KNNRegressor com $k=3$ (Euclidiana)
- KNNRegressor com $k=5$ (Cosseno)
- KNNRegressor com $k=7$ (Euclidiana)
- NormalLinearRegression
O resultado foi visualizado em um Scatter Plot, comparando os valores previstos de cada modelo com os valores reais (y_test) do Performance Index.

4. Manipulação de Imagens
A última etapa consistiu em carregar a imagem mario.png em um array NumPy e realizar transformações básicas usando indexação e fatiamento.

- Transformações Geradas:
1. Original
2. Metade Esquerda
3. Metade Superior
4. Flip Horizontal
5. Flip Vertical
6. Imagem em Escala de Cinza

Todas as 6 imagens foram exibidas em um único grid de visualização (2x3).

- Foi realizado a plotagem de seis funções de ativação comuns e essenciais para introduzir não-linearidades em modelos de Deep Learning.

- Todas as funções são apresentadas em um único gráfico para comparação direta de seus comportamentos.

O gráfico compara o comportamento das seguintes funções no intervalo de $x$ de $-5$ a $5$:
- Sigmoide ($\sigma(x)$): Comprime a entrada para o intervalo $(0, 1)$, ideal para a camada de saída em tarefas de classificação binária.
- Tangente Hiperbólica (Tanh): Comprime a entrada para o intervalo $(-1, 1)$, que muitas vezes é preferível à Sigmoide por ter média em zero.
- ReLU (Rectified Linear Unit): Define a saída como $\max(0, x)$, sendo a função padrão para camadas intermediárias devido à sua eficiência computacional.
- Leaky ReLU: Uma variação da ReLU que permite um gradiente pequeno e não-nulo para entradas negativas ($0.01x$), ajudando a mitigar o problema do "neurônio morrendo" (dying ReLU).
- Swish: Introduzida recentemente, é uma função self-gated dada por $x \cdot \sigma(x)$, conhecida por superar a ReLU em tarefas mais profundas.
- GeLU (Gaussian Error Linear Unit): Uma função de ativação mais moderna e suave, baseada na distribuição cumulativa normal, frequentemente usada em modelos de linguagem como BERT.

