**Machine Learning tips and tricks translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-machine-learning-tips-and-tricks)

<br>

**1. Machine Learning tips and tricks cheatsheet**

&#10230; Machine Learning spiekbriefje

<br>

**2. Classification metrics**

&#10230; Prestatiematen classificatie

<br>

**3. In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

&#10230; Hieronder de belangrijkste maten om de prestatie van een model te evalueren binnen de context van binaire classificatie.

<br>

**4. Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**
]
&#10230; Verwarringsmatrix ― De verwarringsmatrix wordt gebruikt om een meer volledig beeld te hebben van de prestatie van een model. Deze wordt als volgt gedefinieerd:

<br>

**5. [Predicted class, Actual class]**

&#10230; [Voorspelde klasse, Werkelijke klasse]

<br>

**6. Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

&#10230; Belangrijkste maten ― De volgende zijnde meest gebruikte prestatiematen om classificatiemodellen te evalueren:

<br>

**7. [Metric, Formula, Interpretation]**

&#10230; [Maat, Formule, Interpretatie]

<br>

**8. Overall performance of model**

&#10230; Algemene correctheid van het model

<br>

**9. How accurate the positive predictions are**

&#10230; De mate waarin positieve voorspellingen correct zijn

<br>

**10. Coverage of actual positive sample**

&#10230; De mate waarin positieven als dusdanig worden herkend

<br>

**11. Coverage of actual negative sample**

&#10230; De mate waarin negatieven als dusdanig worden herkend

<br>

**12. Hybrid metric useful for unbalanced classes**

&#10230; Een samengestelde maat die nuttig kan zijn bij ongebalanceeerde klasses

<br>

**13. ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

&#10230; ROC ― De *receiver operating characteristic curve*, ook wel ROC-curve, is de grafiek waarin de sensitiviteit tegenover de vals positieve ratio wordt afgebeeld voor verschillende waarden voor het afkappingspunt.

<br>

**14. [Metric, Formula, Equivalent]**

&#10230; [Maat, Formule, Equivalent]

<br>

**15. AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

&#10230; AUC ― *Area under the receiver operating characteristic curve*, ook wel AUC of AUROC, is het gebied onder de ROC zoals in de volgende figuur wordt getoond:

<br>

**16. [Actual, Predicted]**

&#10230; [Werkelijk, Voorspeld]

<br>

**17. Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**

&#10230; Basismaten ― Gegeven een regrossiemodel f zijn dit de meest gebruikte maten om de prestatie van het model te evalueren:

<br>

**18. [Total sum of squares, Explained sum of squares, Residual sum of squares]**

&#10230; [Totale kwadratensom, Verklaarde kwadratensom, Residuele kwadratensom]

<br>

**19. Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

&#10230; Determinatiecoëfficient ― De determinatiecoëfficient, vaak genoteerd als R2 of r2, is een maat van hoe goed de geobserveerde uitkomsten door het model kunnen worden gerepliceerd. Deze is als volgt gedefinieerd:

<br>

**20. Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

&#10230; Belangrijkste maten ― De volgende maten worden het meest gebruikt om de prestatie van regressiemodellen te evalueren, rekening houdend met het aantal variabelen n dat in de modellen wordt opgenomen:

<br>

**21. where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

&#10230; met L de aannemelijkheid en ˆσ2 een schatting van de variantie van iedere respons.

<br>

**22. Model selection**

&#10230; Modelselectie

<br>

**23. Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230; Woordenschat ― Bij het bouwen van een model maken we een onderscheid tussen drie soorten datasets: 

<br>

**24. [Training set, Validation set, Testing set]**

&#10230; [Trainingsset, Validatieset, Testset]

<br>

**25. [Model is trained, Model is assessed, Model gives predictions]**

&#10230; [Model wordt getraind, Model wordt uitgeprobeerd, Model geeft voorspellingen]

<br>

**26. [Usually 80% of the dataset, Usually 20% of the dataset]**

&#10230; [Gewoonlijk 80% vande dataset, Gewoonlijk 20% van de dataset]

<br>

**27. [Also called hold-out or development set, Unseen data]**

&#10230; [Ook bewaar-apart- of ontwikkelingsset genoemd]

<br>

**28. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230; Eénmaal een model gekozen is wordt het getraind op de volledige dataset en getest op de ongeziene testset. Deze worden hieronder weergegeven:

<br>

**29. Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

&#10230; Kruisvalidatie ― Kruisvalidatie, ook wel CV (*cross-validation*), is een methode om aan modelselectie te kunnen doen zonder al te veel af te hangen van de voorhanden data. De verschillende soorten worden opgesomd in onderstaande tabel:

<br>

**30. [Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

&#10230; [Training gebeurt op k-1 delen en de evaluatie op het resterende, Training gebeurt op n-p observaties en de evaluatie op de p resterende]

<br>

**31. [Generally k=5 or 10, Case p=1 is called leave-one-out]**

&#10230; [Gewoonlijk is k=5 of 10, In het geval van p=1 spreken we van *leave one out* of houd-één-uit]

<br>

**32. The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

&#10230; De meestgebruikte methode is k-voudige kruisvalidatie en splitst de trainingdata in k delen om het model te evalueren op 1 deel terwijl de training gebeurt op de k-1 andere delen, dit alles k keer. De fout wordt dan gemiddeld over de k delen en wordt de kruisvalidatiefout genoemd.

<br>

**33. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230; Regularisatie ― Regularisatie probeert te voorkomen dat het model de data gaat *overfitten* en rekent dus af met problemen als hoge variantie. De volgende tabel somt de verschillende vaakgebruikte regularisatietechnieken op:

<br>

**34. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; [Laat coëfficienten krimpen tot 0, Goed om variabelen te selecteren, Maakt coëfficienten kleiner, Afweging tussen variabelen selecteren en kleinere coëfficienten]

<br>

**35. Diagnostics**

&#10230; Probleemherkenning

<br>

**36. Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

&#10230; Vertekening ― De vertekening van een model is het verschil tussen de verwachte voorspelling en de correcte waarden die we proberen te voorspelen voor een gegeven set datapunten.

<br>

**37. Variance ― The variance of a model is the variability of the model prediction for given data points.**

&#10230; Variantie ― De variantie van een model is de variabiliteit van de voorspellingen voor een gegeven set datapunten.

<br>

**38. Bias/variance tradeoff ― The simpler the model, the higher the bias, and the more complex the model, the higher the variance.**

&#10230; Vertekening/variantie-afweging ― Hoe eenvoudiger het model, hoe meer vertekening, en hoe complexer het model, hoe meer variantie.

<br>

**39. [Symptoms, Regression illustration, classification illustration, deep learning illustration, possible remedies]**

&#10230; [Symptomen, Voorbeeld regressie, Voorbeeld classificatie, Voorbeeld deep learning, Mogelijke remedies]

<br>

**40. [High training error, Training error close to test error, High bias, Training error slightly lower than test error, Very low training error, Training error much lower than test error, High variance]**

&#10230; [Hoge trainingfout, Trainingfout dicht tegen de testfout, Veel vertekening, Trainingfout lichtjes lager dan de testfout, Zeer lage trainingfout, Trainingfout veel lager dan testfout, Hoge variantie]

<br>

**41. [Complexify model, Add more features, Train longer, Perform regularization, Get more data]**

&#10230; [Model complexer maken, Meer variabelen toevoegen, Langer laten trainen, Regularisatie toepassen, Meer data verzamelen]

<br>

**42. Error analysis ― Error analysis is analyzing the root cause of the difference in performance between the current and the perfect models.**

&#10230; [Foutenanalyse ― Foutenanalyse bestaat uit het zoeken naar de oorzaken van verschillen in prestatie tussen het huidige model en perfecte modellen.]

<br>

**43. Ablative analysis ― Ablative analysis is analyzing the root cause of the difference in performance between the current and the baseline models.**

&#10230; [Ablatieve analyse ― Ablatieve analyse bestaat uit het zoeken naar de oorzaken van verschillen in prestatie tussen het huidige model en basislijnmodellen.]

<br>

**44. Regression metrics**

&#10230; Prestatiematen regressie

<br>

**45. [Classification metrics, confusion matrix, accuracy, precision, recall, F1 score, ROC]**

&#10230; [Prestatiematen classificatie, verwarringsmatrix, accuraatheid, precisie, sensitiviteit, F1 score, ROC]

<br>

**46. [Regression metrics, R squared, Mallow's CP, AIC, BIC]**

&#10230; [Prestatiematen regressie, R-kwadraat, Mallow's CP, AIC, BIC]

<br>

**47. [Model selection, cross-validation, regularization]**

&#10230; [Modelselectie, kruisvalidatie, regularisatie]

<br>

**48. [Diagnostics, Bias/variance tradeoff, error/ablative analysis]**

&#10230; [Probleemherkenning, Vertekening/variantie-afweging, fouten-/ablatieve analyse]
