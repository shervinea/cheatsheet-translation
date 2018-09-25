**1. Machine Learning tips and tricks cheatsheet**

&#10230;Consells i trucs sobre Aprenentatge Automàtic

<br>

**2. Classification metrics**

&#10230; Mètriques per a classificació

<br>

**3. In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

&#10230; En el context d'una classificació binària, aquestes son les principals mètriques que es important seguir per a evaluar el rendiment del model.

<br>

**4. Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**

&#10230; Matriu de confusió - la matriu de confusió s'utilitza per a tindre una visió més completa a l'evaluar el rendimiento d'un model. Es defineix de la següent forma:

<br>

**5. [Predicted class, Actual class]**

&#10230; [Classe predita, classe real]

<br>

**6. Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

&#10230; Mètriques principals - les següents mètriques s'utilitzen comunment per a evaluar el rendiment dels models de classificació:

<br>

**7. [Metric, Formula, Interpretation]**

&#10230; [Mètrica, Fórmula, Interpretació]

<br>

**8. Overall performance of model**

&#10230; Rendiment general del model

<br>

**9. How accurate the positive predictions are**

&#10230; Com de precises son les prediccions positives?

<br>

**10. Coverage of actual positive sample**

&#10230; Cobertura de la mostra positiva real

<br>

**11. Coverage of actual negative sample**

&#10230; Cobertura de la mostra negativa real

<br>

**12. Hybrid metric useful for unbalanced classes**

&#10230; Mètrica híbrida útil per a classes desbalancejades

<br>

**13. ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

&#10230; ROC ― La Corba característica Operativa del Receptor, també coneguda com ROC, es una representació gràfica de la sensibilitat davant a la especificitat segons varia el llindar. Aquestes mètriques es resumeixen en la taula a continuació:

<br>

**14. [Metric, Formula, Equivalent]**

&#10230; [Mètrica, Fórmula, Interpretació]

<br>

**15. AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

&#10230; AUC ― L'àrea sota la corba Característica Operativa del Receptor, també denotada com AUC o AUROC, es l'àrea sota del ROC, com es mostra en la següent figura:

<br>

**16. [Actual, Predicted]**

&#10230; [Actual, predita]

<br>

**17. Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**

&#10230; Mètriques bàsiques - Donat un model de regresió f, les següents mètriques s'utilitzen comunmente per a evaluar el rendimient del model:

<br>

**18. [Total sum of squares, Explained sum of squares, Residual sum of squares]**

&#10230; [Suma total de quadrats, suma de quadrats explicada, suma residual de quadrats]

<br>

**19. Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

&#10230; Coeficient de determinació: el coeficient de determinació, a sovint denotat com R2 o r2, proporciona una mesura de com de bé els resultats observats son replicats por el model i es defineix de la següent forma:

<br>

**20. Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

&#10230; Mètriques principals: les següents mètriques s'utilitzen comunment per a evaluar el rendiment dels models de regressió, tenint en compte la quantitat de variables n que tenen en compte:

<br>

**21. where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

&#10230; on L es la probabilitat i ˆσ2 es una estimació de la variància associada amb cada resposta.

<br>

**22. Model selection**

&#10230; Selecció de model

<br>

**23. Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230; Vocabulari - al seleccionar un model, distingim 3 parts diferents de les dades que tenim de la següent forma:

<br>

**24. [Training set, Validation set, Testing set]**

&#10230; [Conjunt d'entrenament, Conjunt de validació, Conjunt de prova]

<br>

**25. [Model is trained, Model is assessed, Model gives predictions]**

&#10230; [Model es entrenat, model es evaluat, model dona prediccions]

<br>

**26. [Usually 80% of the dataset, Usually 20% of the dataset]**

&#10230; [Generalment el 80% del conjunt de dades, Generalment el 20% del conjunt de dades]

<br>

**27. [Also called hold-out or development set, Unseen data]**

&#10230; [També anomenat hold-out o conjunt de desenvolupament, Dades no vistes]

<br>

**28. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230; Una vegada que s'ha escollit el model, s'entrena sobre tot el conjunt de dades i es testeja sobre el conjunt de prova no vist. Aquests estan representats en la figura a continuació:

<br>

**29. Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

&#10230; Validació creuada - La validació creuada, també denominada CV, es un mètode que s'utilitza per a seleccionar un model que no confíe excesivament en el conjunt d'entrenament inicial. Els diferents tipus es resumeixen en la taula següent:

<br>

**30. [Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

&#10230; [Entrenament sobre els conjunts k-1 i evaluació en la resta, Entrenament en observacions n-p y evaluació en els p restants

<br>

**31. [Generally k=5 or 10, Case p=1 is called leave-one-out]**

&#10230; [Generalment k = 5 o 10, el cas p = 1 se l'anomena deixant un fora]

<br>

**32. The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

&#10230; El mètodo més comunment utilitzat es denomina validació creuada k-fold i divideix les dades d'entrenament en k conjunts per a validar el model sobre un conjunt mentre s'entrena el model amb els altres k-1 conjunts, tot açò k vegades. L'error després es promèdia sobre els k conjunts i es denomina error de validació creuada.

<br>

**33. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230; Regularització - El procedimient de regularització té com a objetiu evitar que el model es sobreajuste a les dades i, per tant, resol els problemes d'alta variància. La següent taula resum els diferents tipus de tècniques de regularizació comunment utilitzades:

<br>

**34. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; [Redueix els coeficients a 0, Bo per a la selecció de variables, Fa que els coeficients siguen més petits, Compensació entre la selecció de variables i els coeficients petits]

<br>

**35. Diagnostics**

&#10230; Diagnòstico

<br>

**36. Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

&#10230; Biaix - El biaix d'un model es la diferència entre la predicció esperada i el model correcte que tratem de predir per a determinats punts de dades.

<br>

**37. Variance ― The variance of a model is the variability of the model prediction for given data points.**

&#10230; Variància - La variància d'un model es la variabilitat de la predicció del model per a punts de dades donats.

<br>

**38. Bias/variance tradeoff ― The simpler the model, the higher the bias, and the more complex the model, the higher the variance.**

&#10230; Correcció de biaix/variància - Quant més simple es el model, major es el biaix, i quant més complex es el model, major es la variància.

<br>

**39. [Symptoms, Regression illustration, classification illustration, deep learning illustration, possible remedies]**

&#10230; [Símptomes, exemple de regresió, exemple de classificació, exemple d'aprenentatge profund, possibles solucions]

<br>

**40. [High training error, Training error close to test error, High bias, Training error slightly lower than test error, Very low training error, Training error much lower than test error, High variance]**

&#10230; [Error d'entrenament alt, Error d'entrenament prop a l'error de prova, Biaix alt, Error d'entrenament lleugerament inferior a l'error de prova, Error d'entrenament molt baix, Error d'entrenament molt més baix que l'error de prova, Variància alta]

<br>

**41. [Complexify model, Add more features, Train longer, Perform regularization, Get more data]**

&#10230; [Complementar el model, agregar més funcions, entrenar més temps, realitzar regularització, obtindre més dades]

<br>

**42. Error analysis ― Error analysis is analyzing the root cause of the difference in performance between the current and the perfect models.**

&#10230; Anàlisi d'errors - L'anàlisi d'errors analitza la causa arrel de la diferència de rendiment entre els models actuals i perfectes.

<br>

**43. Ablative analysis ― Ablative analysis is analyzing the root cause of the difference in performance between the current and the baseline models.**

&#10230; Anàlisi ablatiu - L'anàlisi ablatiu analitza la causa arrel de la diferència en el rendiment entre els models actuals i de referència.

<br>

**44. Regression metrics**

&#10230; Mètriques de regressió

<br>

**45. [Classification metrics, confusion matrix, accuracy, precision, recall, F1 score, ROC]**

&#10230; [Mètriques de classificació, matriu de confusió, exactitut, precisió, recall, F1 score, ROC]

<br>

**46. [Regression metrics, R squared, Mallow's CP, AIC, BIC]**

&#10230; [Mètriques de regressió, R quadrat, Mallow's CP, AIC, BIC]

<br>

**47. [Model selection, cross-validation, regularization]**

&#10230; [Selecció de model, validació creuada, regularització]

<br>

**48. [Diagnostics, Bias/variance tradeoff, error/ablative analysis]**

&#10230; [Diagnòstic, Correcció Biaix/variància, Anàlisi d'errors/ablatiu]