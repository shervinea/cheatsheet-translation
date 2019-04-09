**1. Machine Learning tips and tricks cheatsheet**

&#10230; Maschinelles Lernen - Tipps und Tricks Spickzettel

<br>

**2. Classification metrics**

&#10230; Definition Klassifikator

<br>

**3. In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

&#10230; Zur Beurteilung und Evaluierung der Leistung eines binären Klassifikators werden folgende Maßzahlen herangezogen: 

<br>

**4. Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**

&#10230; Konfusionsmatrix - Die Konfusionsmatrix gibt Einblick in die allgemeine Leistung des Modelles und wird wie folgt definiert: 

<br>

**5. [Predicted class, Actual class]**

&#10230; [Ermittelte Klasse, Tatsächliche Klasse]

<br>

**6. Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

&#10230; Hauptmaßzahlen - Folgende Hauptmaßzahlen werden zur Beurteilung eines Klassifikators verwendet:

<br>

**7. [Metric, Formula, Interpretation]**

&#10230; [Kennzahl, Formel, Interpretation]

<br>

**8. Overall performance of model**

&#10230; Anteil aller korrekt klassifizierten Daten

<br>

**9. How accurate the positive predictions are**

&#10230; Rate der Korrektheit der positiv klassifizierten Ergebnisse

<br>

**10. Coverage of actual positive sample**

&#10230; Umfang der tatsächlichen positiven Proben

<br>

**11. Coverage of actual negative sample**

&#10230; Umfang der tatsächlichen negativen Proben

<br>

**12. Hybrid metric useful for unbalanced classes**

&#10230; Kombiniertes Maß zur Beurteilung der Güte

<br>

**13. ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

&#10230; ROC -  Die Receiver-Operating-Characteristic-Kurve, auch ROC genannt, stellt die Relation zwischen TPR und FPR dar. Diese werden in der folgenden Tabelle zusammengefasst:

<br>

**14. [Metric, Formula, Equivalent]**

&#10230; [Kennzahl, Formel, Pendant]

<br>

**15. AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

&#10230; AUC - Fläche unter der ROC Kurve, auch AUC/AUROC genannt. Folgende Abbildung verdeutlicht dies:

<br>

**16. [Actual, Predicted]**

&#10230; [Tatsächlich, Ermittelt]

<br> 

**17. Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**

&#10230; Allgemeine Maßzahlen - Für ein Regressionsmodell f werden folgende Verfahren zur Modellvalidierung verwendet:

<br> 

**18. [Total sum of squares, Explained sum of squares, Residual sum of squares]**

&#10230; [Totale Quadratsumme, Erklärte Abweichungsquadratsumme, Residuenquadratsumme]

<br>

**19. Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

&#10230; Bestimmtheitsmaß - Bestimmtheitsmaß, auch R2 oder r2 genannt, ist eine statistische Kennzahl zur Beurteilung der Anpassungsgüte des Modelles und ist wie folgt definiert:

<br>

**20. Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

&#10230; Globale Maßzahl - Folgende Kennzahlen geben die Maße der Anpassung eines Regressionsmodelles an. Folgende Metriken berücksichtigen ebenso die Anzahl der geschätzte Variablen n:

<br>

**21. where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

&#10230; L ist der Likelihood-Wert und ˆσ2 ist die geschätze Varianz per Zielvariable

<br>

**22. Model selection**

&#10230; Modellauswahl

<br>

**23. Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230; Datensatz - Bei der Auswahl des Modells wird der grundlegende Datensatz in drei Datensätze geteilt:

<br>

**24. [Training set, Validation set, Testing set]**

&#10230; [Trainingsdatensatz, Validierungsdatensatz, Testdatensatz]

<br>

**25. [Model is trained, Model is assessed, Model gives predictions]**

&#10230; [Modell ist trainiert, Modell ist evaluiert, Modell prognostiziert]

<br>

**26. [Usually 80% of the dataset, Usually 20% of the dataset]**

&#10230; [Meist 80% des Datensatzes, Meist 20% des Datensatzes]

<br>

**27. [Also called hold-out or development set, Unseen data]**

&#10230; Auch Hold-Out oder Entwicklungsset genannt, Ungesehene Daten]

<br>

**28. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230; Sobald entschieden ist welches Modell verwendet wird, wird dieses mit dem gesamten Datensatz trainiert und mit dem Validierungsdatensatz getestet, wie folgende Abbildung zeigt:

<br>

**29. Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

&#10230; Kreuzvalidierung - Bei der Kreuzvalidierung wird eine Lernmodell ausgewählt welches die Abhängigkeit des Lernmodelles vom initialen Trainingsdatensatz gering hält. Verschiedene Verfahren werden in der folgenden Tabelle erläutert:

<br>

**30. [Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

&#10230; Training mit k-1 Teilmengen und Evaluierung mit den übrigen Daten, Training mit n-p Proben und Evaluierung mit den übrigen Daten]

<br>

**31. [Generally k=5 or 10, Case p=1 is called leave-one-out]**

&#10230; [Meist k=5 oder 10, p=1 wird auch "Leave-One-Out" genannt]

<br>

**32. The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

&#10230; Am häufigsten wird die k-fache Kreuzvalidierung verwendet, welche den Trainingsdatensatz in k-Teilmengen trennt. Das Modell wird jeweils mit der k-te Teilmenge validiert und mit den verbleibenden k-1 Teilmengen trainiert und k-mal wiederholt. Die Gesamtfehlerrate der Kreuzvalidierung ist der Durchschnitt der Einzelfehlerraten der k Einzeldurchläufe.

<br>

**33. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230; Regularisierung - Regularisierung dient der Vermeidung von Überanpassung des Modells an den Trainingsdaten und ist eine Gegenmaßnahme bei hoher Varianz. Folgende Tabelle beschreibt verschiedene Techniken der Regularisierung:

<br>

**34. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; [Annäherung des Koeffizienten an 0, Gut bei Parameterauswahl, Verkleinern des Koeffizienten, Kompromiss zwischen Auswahl der Parameter und kleiner Koeffizient]

<br>

**35. Diagnostics**

&#10230; Modellevaluierung

<br>

**36. Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

&#10230; Verzerrung - Die Verzerrung des Modells ist die Differenz zwischen den erwarteten Prognosen und dem korrektem Modell des verwendeten Datensatzes welches prognostiziert werden soll.

<br>

**37. Variance ― The variance of a model is the variability of the model prediction for given data points.**

&#10230; Varianz - Die Varianz des Modells ist die Schwankung der Modellprognose für den verwendeten Datensatz.

<br>

**38. Bias/variance tradeoff ― The simpler the model, the higher the bias, and the more complex the model, the higher the variance.**

&#10230; Verzerrung-Varianz-Dilemma - Umso einfacher das Modell, desto höher ist die Verzerrung. Umso komplexer das Modell, desto höher ist die Varianz. 

<br>

**39. [Symptoms, Regression illustration, classification illustration, deep learning illustration, possible remedies]**

&#10230; [Symptome, Regressionsabbildung, Klassifikationsabbildung, Deep Learning Beispiel, Mögliche Lösungen]

<br>

**40. [High training error, Training error close to test error, High bias, Training error slightly lower than test error, Very low training error, Training error much lower than test error, High variance]**

&#10230; [Hohe Trainingsfehlerrate, Trainingsfehler nahe am Testfehler, Hohe Verzerrung, Trainingsfehler geringer als Testfehler, Sehr geringer Trainingsfehler, Trainingsfehler erheblich geringer als Testfehler, Hohe Varianz]

<br>

**41. [Complexify model, Add more features, Train longer, Perform regularization, Get more data]**

&#10230; [Modellkomplexität, Ergänzung von Merkmalen, Längere Trainingszeit, Regularisierung, Erweiterung des Datensatzes]

<br>

**42. Error analysis ― Error analysis is analyzing the root cause of the difference in performance between the current and the perfect models.**

&#10230; Fehleranalyse - Bei der Fehleranalyse wird der Ursache der Differenz zwischen der Leistung des korrektem und des derzeitigen Modelles analysiert.

<br>

**43. Ablative analysis ― Ablative analysis is analyzing the root cause of the difference in performance between the current and the baseline models.**

&#10230; Ablative Analyse - Bei der ablativen Analyse wird die Ursache der Differenz zwischen der Leistung der Baseline und des derzeitigen Modelles analysiert.

<br>

**44. Regression metrics**

&#10230; Regressions Metriken

<br>

**45. [Classification metrics, confusion matrix, accuracy, precision, recall, F1 score, ROC]**

&#10230; [Klassifikationsbewertung, Konfusionsmatrix, Treffergenauigkeit, Genauigkeit, Trefferquote, F-Maß, ROC]

<br>

**46. [Regression metrics, R squared, Mallow's CP, AIC, BIC]**

&#10230; [Regressionsbewertung, R Quadrat, Mallow's CP Statistik, AIC, BIC (SBC)]

<br>

**47. [Model selection, cross-validation, regularization]**

&#10230; [Modellauswahl, Kreuzvalidierung, Regularisierung]

<br>

**48. [Diagnostics, Bias/variance tradeoff, error/ablative analysis]**

&#10230; [Diagnose, Verzerrung-Varianz-Dilemma, Fehler/Ablativ Analyse]
