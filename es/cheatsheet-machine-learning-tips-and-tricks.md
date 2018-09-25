**1. Machine learning tips and tricks cheatsheet**

&#10230; Hoja de referencia de consejos y trucos sobre Aprendizaje Automático

<br>

**2. Classification metrics**

&#10230; Métricas para clasificación

<br>

**3. In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

&#10230; En el contexto de una clasificación binaria, estas son las principales métricas que son importantes seguir para evaluar el rendimiento del modelo.

<br>

**4. Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**

&#10230; Matriz de confusión - la matriz de confusión se utiliza para tener una visión más completa al evaluar el rendimiento de un modelo. Se define de la siguiente manera:

<br>

**5. [Predicted class, Actual class]**

&#10230; [Clase predicha, clase real]

<br>

**6. Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

&#10230; Métricas principales - las siguientes métricas se utilizan comúnmente para evaluar el rendimiento de los modelos de clasificación:

<br>

**7. [Metric, Formula, Interpretation]**

&#10230; [Métrica, Fórmula, Interpretación]

<br>

**8. Overall performance of model**

&#10230; Rendimiento general del modelo

<br>

**9. How accurate the positive predictions are**

&#10230; Que tan precisas son las predicciones positivas

<br>

**10. Coverage of actual positive sample**

&#10230; Cobertura de la muestra positiva real

<br>

**11. Coverage of actual negative sample**

&#10230; Cobertura de la muestra negativa real

<br>

**12. Hybrid metric useful for unbalanced classes**

&#10230; Métrica híbrida útil para clases desbalanceadas

<br>

**13. ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

&#10230; ROC ― La curva Característica Operativa del Receptor, también conocida como ROC, es una representación gráfica de la sensibilidad frente a la especificidad según se varía el umbral. Estas métricas se resumen en la tabla a continuación:

<br>

**14. [Metric, Formula, Equivalent]**

&#10230; [Métrica, Fórmula, Interpretación]

<br>

**15. AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

&#10230; AUC ― El área bajo la curva Característica Operativa del Receptor, también conocida como AUC o AUROC, es el área debajo del ROC, como se muestra en la siguiente figura:

<br>

**16. [Actual, Predicted]**

&#10230; [Actual, predicha]

<br>

**17. Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**

&#10230; Métricas básicas - dado un modelo de regresión f, las siguientes métricas se usan comúnmente para evaluar el rendimiento del modelo:

<br>

**18. [Total sum of squares, Explained sum of squares, Residual sum of squares]**

&#10230; [Suma total de cuadrados, suma de cuadrados explicada, suma residual de cuadrados]

<br>

**19. Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

&#10230; Coeficiente de determinación: el coeficiente de determinación, a menudo indicado como R2 o r2, proporciona una medida de lo bien que los resultados observados son replicados por el modelo y se define de la siguiente manera:

<br>

**20. Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

&#10230; Métricas principales: las siguientes métricas se utilizan comúnmente para evaluar el rendimiento de los modelos de regresión, teniendo en cuenta la cantidad de variables n que tienen en cuenta:

<br>

**21. where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

&#10230; donde L es la probabilidad y ˆσ2 es una estimación de la varianza asociada con cada respuesta.

<br>

**22. Model selection**

&#10230; Selección de modelo

<br>

**23. Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230; Vocabulario - al seleccionar un modelo, distinguimos 3 partes diferentes de los datos que tenemos de la siguiente manera:

<br>

**24. [Training set, Validation set, Testing set]**

&#10230; [Conjunto de entrenamiento, Conjunto de validación, Conjunto de prueba]

<br>

**25. [Model is trained, Model is assessed, Model gives predictions]**

&#10230; [Modelo es entrenado, modelo es evaluado, modelo da predicciones]

<br>

**26. [Usually 80% of the dataset, Usually 20% of the dataset]**

&#10230; [Generalmente el 80% del conjunto de datos, Generalmente el 20% del conjunto de datos]

<br>

**27. [Also called hold-out or development set, Unseen data]**

&#10230; [También llamado hold-out o conjunto de desarrollo, Datos no vistos]

<br>

**28. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230; Una vez que se ha elegido el modelo, se entrena sobre todo el conjunto de datos y se testea sobre el conjunto de prueba no visto. Estos están representados en la figura a continuación:

<br>

**29. Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

&#10230; Validación cruzada - la validación cruzada, también denominada CV (por sus siglas en ingles), es un método que se utiliza para seleccionar un modelo que no confíe demasiado en el conjunto de entrenamiento inicial. Los diferentes tipos se resumen en la tabla a continuación:

<br>

**30. [Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

&#10230; [Entrenamiento sobre los conjuntos k-1 y evaluación en el restante, Entrenamiento en observaciones n-p y evaluación en los p restantes

<br>

**31. [Generally k=5 or 10, Case p=1 is called leave-one-out]**

&#10230; [Generalmente k = 5 o 10, el caso p = 1 se llama dejando uno fuera]

<br>

**32. The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

&#10230; El método más comúnmente utilizado se denomina validación cruzada k-fold y divide los datos de entrenamiento en k conjuntos para validar el modelo sobre un conjunto mientras se entrena el modelo en los otros k-1 conjuntos, todo esto k veces. El error luego se promedia sobre los k conjuntos y se denomina error de validación cruzada.

<br>

**33. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230; Regularización - el procedimiento de regularización tiene como objetivo evitar que el modelo se sobreajuste a los datos y, por lo tanto, resuelve los problemas de alta varianza. La siguiente tabla resume los diferentes tipos de técnicas de regularización comúnmente utilizadas:

<br>

**34. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; [Reduce los coeficientes a 0, Bueno para la selección de variables, Hace que los coeficientes sean más pequeños, Compensación entre la selección de variables y los coeficientes pequeños]

<br>

**35. Diagnostics**

&#10230; Diagnóstico

<br>

**36. Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

&#10230; Sesgo - el sesgo de un modelo es la diferencia entre la predicción esperada y el modelo correcto que tratamos de predecir para determinados puntos de datos.

<br>

**37. Variance ― The variance of a model is the variability of the model prediction for given data points.**

&#10230; Varianza - la varianza de un modelo es la variabilidad de la predicción del modelo para puntos de datos dados.

<br>

**38. Bias/variance tradeoff ― The simpler the model, the higher the bias, and the more complex the model, the higher the variance.**

&#10230; Corrección de sesgo/varianza - cuanto más simple es el modelo, mayor es el sesgo, y cuanto más complejo es el modelo, mayor es la varianza.

<br>

**39. [Symptoms, Regression illustration, classification illustration, deep learning illustration, possible remedies]**

&#10230; [Síntomas, ejemplo de regresión, ejemplo de clasificación, ejemplo de aprendizaje profundo, posibles soluciones]

<br>

**40. [High training error, Training error close to test error, High bias, Training error slightly lower than test error, Very low training error, Training error much lower than test error, High variance]**

&#10230; [Error de entrenamiento alto, Error de entrenamiento cercano al error de prueba, Sesgo alto, Error de entrenamiento ligeramente inferior al error de prueba, Error de entrenamiento muy bajo, Error de entrenamiento mucho más bajo que el error de prueba, Varianza alta]

<br>

**41. [Complexify model, Add more features, Train longer, Perform regularization, Get more data]**

&#10230; [Incrementar la complejidad del modelo, agregar más funciones, entrenar más tiempo, realizar la regularización, obtener más datos]

<br>

**42. Error analysis ― Error analysis is analyzing the root cause of the difference in performance between the current and the perfect models.**

&#10230; Análisis de errores - el análisis de errores analiza la causa raíz de la diferencia de rendimiento entre los modelos actuales y perfectos.

<br>

**43. Ablative analysis ― Ablative analysis is analyzing the root cause of the difference in performance between the current and the baseline models.**

&#10230; Análisis ablativo - el análisis ablativo analiza la causa raíz de la diferencia en el rendimiento entre los modelos actuales y de referencia.

<br>
