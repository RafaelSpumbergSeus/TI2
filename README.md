README: Análise e Modelagem de Performance de Estudantes

Este projeto cobre o processamento, limpeza, análise exploratória e modelagem de um conjunto de dados sobre a performance de estudantes, culminando na implementação de dois algoritmos de regressão (KNN e Regressão Linear) do zero.

1. Preparação e Limpeza de DadosO arquivo Student_Performance_sujo.csv foi processado para garantir a qualidade dos dados.

1.1 Limpeza e Feature Engineering

As seguintes operações foram realizadas no dataset: 
    - Carregamento e Limpeza Inicial: O arquivo foi carregado e os nomes das colunas foram padronizados.
    - Remoção de Duplicatas: Foram eliminados os registros duplicados.
    - Remoção de Outliers: Registros com valores em Sample_Question_Papers_Practiced acima do percentil 80 foram removidos.
    - Tratamento de Faltantes: Valores NaN nas colunas numéricas foram preenchidos com a média da respectiva coluna.
    - Conversão Categórica: A coluna Extracurricular_Activities foi binarizada: Yes $\to$ 1 e No $\to$ 0.
    - Criação de Feature: Foi criada a coluna practice_per_hour (razão entre provas práticas e horas estudadas).
    - Normalização Z-score: Todas as colunas de entrada (exceto Performance_Index) foram normalizadas usando Z-score ($z = \frac{x - \mu}{\sigma}$).

Shape Final do Dataset Limpo: (9904, 7)

1.2 Gráfico de Funções de Ativação

Um gráfico comparativo foi gerado para visualizar as funções de ativação mais comuns em Redes Neurais, usando 1000 pontos no intervalo $[-5, 5]$.
    - Funções Incluídas: Sigmoide, ReLU, Tangente Hiperbólica (Tanh), GeLU, Leaky ReLU e Swish.

2. Visualização de Dados (EDA)

Após a limpeza, foram gerados gráficos para entender a distribuição e a correlação dos dados.

2.1. Distribuição da Variável Alvo

Um Histograma foi criado para a variável alvo Performance_Index após a aplicação da transformação logarítmica ($\text{log}(1+x)$), visando uma distribuição mais próxima do ideal para modelos de Machine Learning.

