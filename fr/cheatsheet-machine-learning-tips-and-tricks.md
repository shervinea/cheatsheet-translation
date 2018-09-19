**1. Machine Learning tips and tricks cheatsheet**

&#10230; Pense-bête de petites astuces de Machine Learning

<br>

**2. Classification metrics**

&#10230; Indicateurs dans le contexte de la classification

<br>

**3. In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

&#10230; Dans le contexte de la classification binaire, voici les principaux indicateurs à surveiller pour évaluer la performance d'un modèle.

<br>

**4. Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**

&#10230; Matrice de confusion ― Une matrice de confusion est utilisée pour avoir une image complète de la performance d'un modèle. Elle est définie de la manière suivante :

<br>

**5. [Predicted class, Actual class]**

&#10230; [Classe prédite, classe vraie]

<br>

**6. Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

&#10230; Indicateurs principaux ― Les indicateurs suivants sont communément utilisés pour évaluer la performance des modèles de classification :

<br>

**7. [Metric, Formula, Interpretation]**

&#10230; [Indicateur, Formule, Interprétation]

<br>

**8. Overall performance of model**

&#10230; Performance globale du modèle

<br>

**9. How accurate the positive predictions are**

&#10230; À quel point les prédictions positives sont précises

<br>

**10. Coverage of actual positive sample**

&#10230; Couverture des observations vraiment positives

<br>

**11. Coverage of actual negative sample**

&#10230; Couverture des observations vraiment négatives

<br>

**12. Hybrid metric useful for unbalanced classes**

&#10230; Indicateur hybride utilisé pour les classes non-balancées

<br>

**13. ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

&#10230; Courbe ROC - La fonction d'efficacité du récepteur, plus fréquemment appelée courbe ROC (de l'anglais *Receiver Operating Curve*), est une courbe représentant le taux de *True Positives* en fonction de taux de *False Positives* et obtenue en faisant varier le seuil. Ces indicateurs sont résumés dans le tableau suivant :

<br>

**14. [Metric, Formula, Equivalent]**

&#10230; [Indicateur, Formule, Equivalent]

<br>

**15. AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

&#10230; AUC ― L'aire sous la courbe ROC, aussi notée AUC (de l'anglais *Area Under the Curve*) ou AUROC (de l'anglais *Area Under the ROC*), est l'aire sous la courbe ROC comme le montre la figure suivante :

<br>

**16. [Actual, Predicted]**

&#10230; [Vraie, Prédite]

<br>

**17. Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**

&#10230; Indicateurs de base ― Étant donné un modèle de régression f, les indicateurs suivants sont communément utilisés pour évaluer la performance d'un modèle :

<br>

**18. [Total sum of squares, Explained sum of squares, Residual sum of squares]**

&#10230; [Somme des carrés totale, Somme des carrés expliquée, Somme des carrés résiduelle]

<br>

**19. Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

&#10230; Coefficient de détermination ― Le coefficient de détermination, souvent notée R2 ou r2, donne une mesure sur la qualité du modèle et est tel que :

<br>

**20. Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

&#10230; Indicateurs principaux ― Les indicateurs suivants sont communément utilisés pour évaluer la performance des modèles de régression, en prenant en compte le nombre de variables n qu'ils prennent en considération :

<br>

**21. where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

&#10230; où L est la vraisemblance et ˆσ2 est une estimation de la variance associée à chaque réponse.

<br>

**22. Model selection**

&#10230; Sélection de modèle

<br>

**23. Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230; Vocabulaire ― Lors de la sélection d'un modèle, on divise les données en 3 différentes parties comme suit :

<br>

**24. [Training set, Validation set, Testing set]**

&#10230; [Ensemble d'apprentissage, Ensemble de validation, Ensemble d'évaluation]

<br>

**25. [Model is trained, Model is assessed, Model gives predictions]**

&#10230; [Modèle est entrainé, Moèle est évalué, Modèle donne des prédictions]

<br>

**26. [Usually 80% of the dataset, Usually 20% of the dataset]**

&#10230; [Normalement 80% du dataset, Normalement 20% du dataset]

<br>

**27. [Also called hold-out or development set, Unseen data]**

&#10230; [Aussi appelé hold-out ou development set, Données jamais vues]

<br>

**28. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230; Une fois que le modèle a été choisi, il est trainé sur le dataset entier et testé sur l'ensemble d'évaluation jamais vu. Ces derniers sont représentés dans la figure ci-dessous :

<br>

**29. Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

&#10230; Validation croisée ― La validation croisée, aussi notée CV (de l'anglais *Cross-Validation*), est une méthode qui est utilisée pour sélectionner un modèle qui ne s'appuie pas trop sur l'ensemble d'apprentissage de départ. Les différents types de validation croisée rencontrés sont récapitulés dans le tableau ci-dessous :

<br>

**30. [Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

&#10230; [Apprentissage sur k-1 folds et évaluation sur le fold restant, Apprentissage sur n-p observations et évaluation sur les p restantes]

<br>

**31. [Generally k=5 or 10, Case p=1 is called leave-one-out]**

&#10230; [Généralement k=5 ou 10, Cas p=1 est appelé leave-one-out]

<br>

**32. The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

&#10230; La méthode la plus utilisée est appelée validation croisée k-fold et partage l'ensemble d'apprentissage en k folds, de manière à valider le modèle sur un fold tout en entrainant le modèle sur les k-1 autres folds, tout ceci k fois. L'erreur est alors moyennée sur k folds et est appelée erreur de validation croisée.

<br>

**33. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230; Régularisation ― La procédure de régularisation a pour but d'éviter que le modèle ne surapprenne (en anglais *overfit*) les données et ainsi vise à régler les problèmes de grande variance. Le tableau suivant récapitule les différentes techniques de régularisation communément utilisées.

<br>

**34. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; [Réduit les coefficients à 0, Bon pour la sélection de variables, Rend les coefficients plus petits, Compromis entre la selection de variables et la réduction de coefficients]

<br>

**35. Diagnostics**

&#10230; Diagnostics

<br>

**36. Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

&#10230; Biais ― Le biais d'un modèle est la différence entre l'espérance de la prédiction et du modèle correct pour lequel on essaie de prédire pour des observations données.

<br>

**37. Variance ― The variance of a model is the variability of the model prediction for given data points.**

&#10230; Variance ― La variance d'un modèle est la variabilité des prédictions d'un modèle pour des observations données.

<br>

**38. Bias/variance tradeoff ― The simpler the model, the higher the bias, and the more complex the model, the higher the variance.**

&#10230; Compromis biais/variance ― Plus le modèle est simple, plus le biais est grand et plus le modèle est complexe, plus la variance est grande.

<br>

**39. [Symptoms, Regression illustration, classification illustration, deep learning illustration, possible remedies]**

&#10230; [Symptômes, Illustration dans le cas de la régression, Illustration dans le cas de la classification, Illustration dans le cas de l'apprentissage profond, remèdes possibles]

<br>

**40. [High training error, Training error close to test error, High bias, Training error slightly lower than test error, Very low training error, Training error much lower than test error, High variance]**

&#10230; [Erreur d'apprentissage élevée, Erreur d'apprentissage proche de l'erreur d'évaluation, Biais élevé, Erreur d'apprentissage légèrement inférieure à l'erreur d'évaluation, Erreur d'apprentissage très faible, Erreur d'apprentissage beaucoup plus faible que l'erreur d'évaluation, Variance élevée]

<br>

**41. [Complexify model, Add more features, Train longer, Perform regularization, Get more data]**

&#10230; [Complexifier le modèle, Ajouter plus de variables, Laisser l'entrainement pendant plus de temps, Effectuer une régularisation, Avoir plus de données]

<br>

**42. Error analysis ― Error analysis is analyzing the root cause of the difference in performance between the current and the perfect models.**

&#10230; Analyse de l'erreur ― L'analyse de l'erreur consiste à analyser la cause première de la différence en performance entre le modèle actuel et le modèle parfait.

<br>

**43. Ablative analysis ― Ablative analysis is analyzing the root cause of the difference in performance between the current and the baseline models.**

&#10230; Analyse ablative ― L'analyse ablative consiste à analyser la cause première de la différence en performance entre le modèle actuel et le modèle de base.

<br>

**44. Regression metrics**

&#10230;
