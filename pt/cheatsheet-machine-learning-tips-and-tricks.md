**1. Machine Learning tips and tricks cheatsheet**

&#10230; Dicas e Truques de Aprendizado de Máquina

<br>

**2. Classification metrics**

&#10230; Métricas de Classificação

<br>

**3. In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

&#10230; Em um contexto de classificação binária, essas são as principais métricas que são importantes acompanhar para avaliar a desempenho do modelo.

<br>

**4. Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**

&#10230; Matriz de Confusão - A matriz de confusão é usada para termos uma figura mais completa quando estamos avaliando o desempenho de um modelo. Ela é definida conforme a seguir:

<br>

**5. [Predicted class, Actual class]**

&#10230; [Classe prevista, classe real]

<br>

**6. Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

&#10230; Principais métricas - As seguintes métricas são comumente usadas para avaliar o desempenho de modelos de classificação. 

<br>

**7. [Metric, Formula, Interpretation]**

&#10230; [Métrica, Fórmula, Interpretação]

<br>

**8. Overall performance of model**

&#10230; Desempenho geral do modelo

<br>

**9. How accurate the positive predictions are**

&#10230; Quão preciso as predições positivas são

<br>

**10. Coverage of actual positive sample**

&#10230; Cobertura da amostra positiva real

<br>

**11. Coverage of actual negative sample**

&#10230; Cobertura da amostra negativa real

<br>

**12. Hybrid metric useful for unbalanced classes**

&#10230; 

<br>

**13. ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

&#10230;

<br>

**14. [Metric, Formula, Equivalent]**

&#10230; [Métrica, Fórmula, Equivalente]

<br>

**15. AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

&#10230;

<br>

**16. [Actual, Predicted]**

&#10230; [Real, Previsto]

<br>

**17. Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**

&#10230; Métricas básicas - Dado um modelo de regresão f, as seguintes métricas são geralmente utilizadas para avaliar o desempenho do modelo:

<br>

**18. [Total sum of squares, Explained sum of squares, Residual sum of squares]**

&#10230;

<br>

**19. Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

&#10230; Coeficiente de determinação - O coeficiente de determinação, frequentemente escrito como R2 ou r2, fornece uma medida de quão bem os resultados observados são replicados pelo modelo e é definido conforme o seguinte:

<br>

**20. Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

&#10230; Principais métricas - A seguintes métricas são comumente utilizadas para avaliar o desempenho de modelos de regressão, levando em conta o número de variáveis n que eles consideram:

<br>

**21. where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

&#10230; onde L é a probabilidade e ˆσ2 é a estimativa da variância associada com cada resposta.

<br>

**22. Model selection**

&#10230; Seleção de Modelo

<br>

**23. Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230; Vocabulário - Ao selecionar um modelo, nós distinguimos 3 diferentes partes dos dados que temos conforme a seguir:

<br>

**24. [Training set, Validation set, Testing set]**

&#10230; [Conjunto de treino, Conjunto de validação, Conjunto de Teste]

<br>

**25. [Model is trained, Model is assessed, Model gives predictions]**

&#10230; [Modelo é treinado, Modelo é avaliado, Modelo fornece previsões]

<br>

**26. [Usually 80% of the dataset, Usually 20% of the dataset]**

&#10230; [Geralmente 80% do conjunto de dados, Geralmente 20% do conjunto de dados]

<br>

**27. [Also called hold-out or development set, Unseen data]**

&#10230; [Também chamado de hold-out ou conjunto de desenvolvimento, Dados não vistos]

<br>

**28. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230; Uma vez que o modelo é escolhido, ele é treinado no conjunto inteiro de dados e testado no conjunto de dados de testes não vistos. São representados na figura abaixo:

<br>

**29. Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

&#10230; Validação cruzada - Validação cruzada, também chamada de CV (Cross-validation), é um método utilizado para selecionar um modelo que não depende muito do conjunto de treinamento inicial. Os diferente tipos são resumidos na tabela abaixo:  

<br>

**30. [Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

&#10230;

<br>

**31. [Generally k=5 or 10, Case p=1 is called leave-one-out]**

&#10230; [Geralmente k=5 ou 10, caso p=1 é chamado deixe-um-fora]

<br>

**32. The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

&#10230;

<br>

**33. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230;

<br>

**34. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230;

<br>

**35. Diagnostics**

&#10230; Diagnóstico

<br>

**36. Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

&#10230;

<br>

**37. Variance ― The variance of a model is the variability of the model prediction for given data points.**

&#10230; Variância - A variância de um modelo é a variabilidade da previsão do modelo para determinados pontos de dados.

<br>

**38. Bias/variance tradeoff ― The simpler the model, the higher the bias, and the more complex the model, the higher the variance.**

&#10230;

<br>

**39. [Symptoms, Regression illustration, classification illustration, deep learning illustration, possible remedies]**

&#10230;

<br>

**40. [High training error, Training error close to test error, High bias, Training error slightly lower than test error, Very low training error, Training error much lower than test error, High variance]**

&#10230;

<br>

**41. [Complexify model, Add more features, Train longer, Perform regularization, Get more data]**

&#10230;

<br>

**42. Error analysis ― Error analysis is analyzing the root cause of the difference in performance between the current and the perfect models.**

&#10230;

<br>

**43. Ablative analysis ― Ablative analysis is analyzing the root cause of the difference in performance between the current and the baseline models.**

&#10230;

<br>

**44. Regression metrics**

&#10230; Métricas de Regressão

<br>

**45. [Classification metrics, confusion matrix, accuracy, precision, recall, F1 score, ROC]**

&#10230; 

<br>

**46. [Regression metrics, R squared, Mallow's CP, AIC, BIC]**

&#10230; [Métricas de Regressão, R Quadrado, Mallow's CP, AIC, BIC]

<br>

**47. [Model selection, cross-validation, regularization]**

&#10230; [Seleção de Modelo, Validação Cruzada, Regularização]

<br>

**48. [Diagnostics, Bias/variance tradeoff, error/ablative analysis]**

&#10230;
