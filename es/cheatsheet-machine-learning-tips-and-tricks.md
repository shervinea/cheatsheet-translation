**1. Machine Learning tips and tricks cheatsheet**

&#10230; Hoja de trucos y ayudas de aprendizaje automático

<br>

**2. Classification metrics**

&#10230; Metricas de clasificación

<br>

**3. In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

&#10230; En el contexto de una clasificacion binaría, estas son las principales metricas que se tienen que checar para evaluar el desempeño del modelo

<br>

**4. Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**

&#10230; Matriz de confusión ― La matriz de confusión es usada para tener un mejor panorama cuandpo se evalua el desempeño del modelo. Se define como:

<br>

**5. [Predicted class, Actual class]**

&#10230; [Clase predecida, Clase Real]

<br>

**6. Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

&#10230; Métricas principales ― Las siguientes métricas son comumente usadas para evaluar el desempeño de modelos clasificatorios
 
<br>

**7. [Metric, Formula, Interpretation]**

&#10230; [Métrica, Fórmula, Interpretación]*

<br>

**8. Overall performance of model**

&#10230; Desempeño en general del modelo

<br>

**9. How accurate the positive predictions are**

&#10230; ¿Qué tan precisan son las precisiones positivas?

<br>

**10. Coverage of actual positive sample**

&#10230; Covertura de las pruebas positivas

<br>

**11. Coverage of actual negative sample**

&#10230; Covertura de las pruebas negativas

<br>

**12. Hybrid metric useful for unbalanced classes**

&#10230; Métrica híbrida útil para clases no balanceadas

<br>

**13. ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

&#10230; Curva ROC ― La curva receptora de operaciones, por sus siglás en inglés ROC, es la gráfica de los Verdaderos Valores Positivos (TPR) contra los Valores Falsos Positivos (FPR) variando el umbral. Estas métricas son englobadas en la tabla siguiente: 

<br>

**14. [Metric, Formula, Equivalent]**

&#10230; [Métrica, Fórmula, Equivalente]

<br>

**15. AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

&#10230; AUC ― Área bajo la curva ROC, también llamda AUC o AUROC, es la área debajo de la curva ROC como se muestra en la siguiente figura:

<br>

**16. [Actual, Predicted]**

&#10230; [Valor actual, Predecido]

<br>

**17. Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**

&#10230; Métricas básicas ― Dado un modelo de regresión f, las siguientes métricas son comunmente usadas para evaluar el desempeño del modelo:

<br>

**18. [Total sum of squares, Explained sum of squares, Residual sum of squares]**

&#10230; [Suma total de los cuadrados, Explicaicón de la suma de cuadrados, Residio de la suma de cuadrados]

<br> 

**19. Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

&#10230; Coefficientes de determinante ― El coeficiente de determinante, usualmente llamado R2 o r2, provee una medida 

<br>

**20. Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

&#10230; Métricas principales ― Las siguientes métricas son comunmente usadas para evaluar el desempeño de los modelos de regresión, tomando el número de variables n que son consideradas:

<br>

**21. where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

&#10230; donde L es la posibilidad y ˆσ2 es una estimación de varianza asociada a cada respuesta.

<br>

**22. Model selection**

&#10230; Selección de modelo

<br>

**23. Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230; Vocabulario ― Cuando se selección un modelo, se tienen 3 diferentes partes de la información como se muestra:

<br>

**24. [Training set, Validation set, Testing set]**

&#10230; [conjunto de entrenamiento, conjunto de validación, conjunto de prueba]

<br>

**25. [Model is trained, Model is assessed, Model gives predictions]**

&#10230; [Entrenamiento de modelo, Evaluación de modelo, Predicción]

<br>

**26. [Usually 80% of the dataset, Usually 20% of the dataset]**

&#10230; [Usualmente 80% del dataset, Usualmente 20% del dataset]

<br>

**27. [Also called hold-out or development set, Unseen data]**

&#10230; [También llamado resistencia o conjunto de desarrollo, información no antes vista]

<br>

**28. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230; Una vez que el modelo se ha escogido, se entrena sobre todo el dataset y se prueba en información no antes vista. Esto se representa en la figura siguiente:

<br>

**29. Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

&#10230; Validación cruzada ― También nombrada CV, es un método que se usa para seleccionar un modelo que no se basa demasaido en el test de entrenamiento inicial. Los diferentes tipos estan englobados en la siguiente tabla:

<br>

**30. [Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

&#10230; [Entrenamiento en k-1 iteraciones y evalua en el restante, Entrenamiento en n-p observaciones y evaluación en las p restantes]

<br>

**31. [Generally k=5 or 10, Case p=1 is called leave-one-out]**

&#10230; [Generalmente k=5 o 10, Caso p=1 es llamado dejar-uno-fuera]

<br>

**32. The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

&#10230; El método mas usado es llamado Validación Cruzada k-iteraciones, que divide el set de entrenamiento en k subconjuntos y valida el modelo en 1 subconjunto mientra entrena el modelo en k-1 subconjuntos, todo esto k veces. El error es el promedio entre las k itreaciones y es nombrado error de validación cruzaa.

<br>

**33. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230; Regularización ― La regularizacion trata de prevenir que el modelo sobreestime la información, por tanto se ocupa de problemas de varianza. La siguiente tabla engloba las diferentes tipos y técnicas más usada de regularización:  

<br>

**34. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; [Minimiza los coeficientes a 0, Bueno para selección de variable, Disminuye los coeficientes, Compensa entre variable de slección y coeficientes pequeño]

<br>

**35. Diagnostics**

&#10230; Diagnosticos

<br>

**36. Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

&#10230; Sesgo ― El sesgo de un modelo es la diferencia entre el valor de predicción esperado y el crrecto valor del model que tratamos de predecir para la información datos

<br>

**37. Variance ― The variance of a model is the variability of the model prediction for given data points.**

&#10230; Varianza ― La varianza de un modelo es la variabilidad de la predicción del modelo para la información dada.

<br>

**38. Bias/variance tradeoff ― The simpler the model, the higher the bias, and the more complex the model, the higher the variance.**

&#10230; Sesgo/compensación de varianza ― A modelo más simple mayor sesgo, mientras más complejo el modelo mayor varianza.

<br>

**39. [Symptoms, Regression illustration, classification illustration, deep learning illustration, possible remedies]**

&#10230; [Síntomas, Figura de regresión, Figura de clasificación, Figura de aprendizaje profundo, Posibles soluciones]

<br>

**40. [High training error, Training error close to test error, High bias, Training error slightly lower than test error, Very low training error, Training error much lower than test error, High variance]**

&#10230; [Alto error de entrenamiento, Error de entrenamiento cercado al error de prueba, Alto sesgo, Error de entrenamiento menor que el error de prueba, Poco error de entrenamiento, Error de entrenamiento mucho menor que error de prueba, Alta varianza]

<br>

**41. [Complexify model, Add more features, Train longer, Perform regularization, Get more data]**

&#10230; [Modelo de complejidad, Añadir más características, Entrenar más tiempo, Hacer regularización, Obtener más datos]

<br>

**42. Error analysis ― Error analysis is analyzing the root cause of the difference in performance between the current and the perfect models.**

&#10230; Error de análisis ― El erro de análisis es analizar la causa raiz de la diferencia en el desempeño del modelo actual y el perfecto. 

<br>

**43. Ablative analysis ― Ablative analysis is analyzing the root cause of the difference in performance between the current and the baseline models.**

&#10230; Análisis Ablativo - Es analizar la causa raiz de la diferencia entre el modelo actual y el base.

<br>
