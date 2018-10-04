**1. Machine Learning tips and tricks cheatsheet**

&#10230; Resumo de dicas e truques de Aprendizado de Máquina

<br>

**2. Classification metrics**

&#10230; Métricas de classificação

<br>

**3. In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

&#10230; Em um contexto de uma classificação binária, existem algumas métricas principais que são importantes de saber para avaliar a performance de um modelo.

<br>

**4. Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**

&#10230; Matriz de confusão - A matriz de confusão é usada para ter um panorama mais completo ao se verificar a performance de um modelo. Ela é definida como:

<br>

**5. [Predicted class, Actual class]**

&#10230; [Classe prevista, Classe real]

<br>

**6. Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

&#10230; Métricas principais - As métricas a seguir são comumente usadas para avaliar a performance de modelos de classificação:

<br>

**7. [Metric, Formula, Interpretation]**

&#10230; [Métrica, Fórmula, Interpretação]

<br>

**8. Overall performance of model**

&#10230; Desempenho geral do modelo

<br>

**9. How accurate the positive predictions are**

&#10230; Quão precisas são as predições positivas:

<br>

**10. Coverage of actual positive sample**

&#10230; Cobertura da amostra positiva real

<br>

**11. Coverage of actual negative sample**

&#10230; Cobertura da amostra negativa real

<br>

**12. Hybrid metric useful for unbalanced classes**

&#10230; Métrica híbrida para classes desbalanceadas

<br>

**13. ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

&#10230; ROC - A curva de operação do receptor, também conhecida como ROC, é o gráfico de TPR versus FPR variando o limiar. Essas métricas são resumidas na tabela abaixo:

<br>

**14. [Metric, Formula, Equivalent]**

&#10230; [Métrica, Fórmula, Equivalente]

<br>

**15. AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

&#10230; AUC - A área sob a curva de operação de recepção, também conhecida como AUC ou AUROC, é a área abaixo do ROC, conforme mostrado na figura a seguir:

<br>

**16. [Actual, Predicted]**

&#10230; [Real, Previsto]

<br>

**17. Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**

&#10230; Métricas básicas - Dado um modelo de regressão f, as seguintes métricas são normalmente usadas para verificar a performance do modelo:

<br>

**18. [Total sum of squares, Explained sum of squares, Residual sum of squares]**

&#10230; [Total da soma de quadrados, Soma explicada de quadrados, Soma residual de quadrados]

<br>

**19. Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

&#10230; Coeficiente de determinação - O coeficiente de determinação, denotado como R2 ou r2, fornece uma medida de quão bem os resultados observados são replicados pelo modelo e são definidos como:

<br>

**20. Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

&#10230; Métricas principais - As seguintes métricas são normalmente usadas para verificar a performance de um modelo de regressão, levando em conta o número de variáveis n que elas levaram em consideração:

<br>

**21. where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

&#10230; onde L é a probabilidade e ˆσ2 é uma estimativa da variância associada com cada respostas.

<br>

**22. Model selection**

&#10230; Seleção de modelos

<br>

**23. Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230;

<br>

**24. [Training set, Validation set, Testing set]**

&#10230; [Conjunto de treinamento, Conjunto de validação, Conjunto de teste]

<br>

**25. [Model is trained, Model is assessed, Model gives predictions]**

&#10230; [Modelo é treinado, Modelo é verificado, Modelo retorna a predição]

<br>

**26. [Usually 80% of the dataset, Usually 20% of the dataset]**

&#10230; [Normalmente 80% do conjunto de dados, Normalmente 20% do conjunto de dados]

<br>

**27. [Also called hold-out or development set, Unseen data]**

&#10230; [Também chamado de hold-out ou conjunto de desenvolvimento, Dados não vistos]

<br>

**28. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230; Uma vez que o modelo foi escolhido, ele é trrinado com um conjunto de dados inteiro e testado com um conjunto de teste não visto. Esses são representados com a figura a seguir:

<br>

**29. Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

&#10230; Validação cruzada (Cross-validation) - A validação cruzada, também denominada CV, é um método que é usado para selecionar um modelo que não confia muito no conjunto de treinamento inicial. Os diferentes tipos estão resumidos na tabela abaixo:

<br>

**30. [Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

&#10230; [Treinamento em conjuntos k-1 e verificação no conjunto remanescenete, Treinamento em observações n-p e verificação nas observações p remanescentes]

<br>

**31. [Generally k=5 or 10, Case p=1 is called leave-one-out]**

&#10230; [Geralmente k=5 ou 10, O caso p=1 é chamado deixe-um-fora (leave-one-out)]

<br>

**32. The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

&#10230; O método mais usado normalmente é chamado de validação cruada k-fold e divide os dados de treinamento em k sub-conjuntos (dobras) para validar o modelo e um conjunto (dobra) para treinar o modelo nos outros k-1 conjuntos, tudo isso k vezes.

<br>

**33. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230; Regularização - O procedimento de regularização para evitar que o modelo sobre-ajuste (overfit) os dados e, portanto, lida com questões de altas variâncias.  A tabela a seguir resume os diferentes tipos de técnicas de regularização mais usadas:

<br>

**34. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; [Diminui coeficientes para 0, Boa seleção de variáveis, Faz com que os coeficientes sejam memores, Faz um balanço entre seleção de variáveis e coeficientes menores]

<br>

**35. Diagnostics**

&#10230; Diagnósticos

<br>

**36. Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

&#10230; Viés - O viés de um modelo é a diferença entre a previsão experada e o modelo corretos que estamos tentando prever para um determinado conjunto de dados.

<br>

**37. Variance ― The variance of a model is the variability of the model prediction for given data points.**

&#10230; Variância - A variância de um modelo é a variabilidade do modelo de predição para um determinado conjunto de dados.

<br>

**38. Bias/variance tradeoff ― The simpler the model, the higher the bias, and the more complex the model, the higher the variance.**

&#10230; Balanço viés/variância - Quanto mais simples o modelo, maior o viés, e quanto mais complexo o modelo, maior a variância.

<br>

**39. [Symptoms, Regression illustration, classification illustration, deep learning illustration, possible remedies]**

&#10230; [Sintomas, Ilustração de regressão, Ilustração de classificação, Ilustração de aprendizagem profunda (deep learning), possíveis soluções]

<br>

**40. [High training error, Training error close to test error, High bias, Training error slightly lower than test error, Very low training error, Training error much lower than test error, High variance]**

&#10230; [Alto erro de treino, Erro de treino próximo ao erro de teste, Alto viés, Erro de treino ligeiramente menor que o erro de teste, Erro de treino muito baixo, Erro de treino muito menor que erro de teste, Alta variância]

<br>

**41. [Complexify model, Add more features, Train longer, Perform regularization, Get more data]**

&#10230; [Modelo complexify, Adicionar mais features, Treinar por mais tempo, Regularizar performance, Conseguir mais dados]

<br>

**42. Error analysis ― Error analysis is analyzing the root cause of the difference in performance between the current and the perfect models.**

&#10230; Erro de análise - Erro de análise analiza a causa raiz da diferença de performance entre o modelo atual e o modelo perfeito.

<br>

**43. Ablative analysis ― Ablative analysis is analyzing the root cause of the difference in performance between the current and the baseline models.**

&#10230; Análise ablativa - A análise ablativa está analisando a causa raiz da diferença de desempenho entre o modelo atual e o modelo de base (baseline).

<br>

**44. Regression metrics**

&#10230; Métricas de regressão

<br>

**45. [Classification metrics, confusion matrix, accuracy, precision, recall, F1 score, ROC]**

&#10230; [Métricas de classificação, matriz de confusão, acurácia, precisão, revocação (recall), F1 score, ROC]

<br>

**46. [Regression metrics, R squared, Mallow's CP, AIC, BIC]**

&#10230; [Métricas de rgessão, R quadrado, CP de Mallow, AIC, BIC]

<br>

**47. [Model selection, cross-validation, regularization]**

&#10230; [Seleção de modelos, validação cruzada, regularização]

<br>

**48. [Diagnostics, Bias/variance tradeoff, error/ablative analysis]**

&#10230; [Diagnóstico, Balanço viés/variância, Análise de erro/ablativa]
