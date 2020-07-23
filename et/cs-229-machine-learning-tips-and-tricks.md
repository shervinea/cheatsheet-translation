**Machine Learning tips and tricks translation** [[webpage]](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-machine-learning-tips-and-tricks)

<br>

**1. Machine Learning tips and tricks cheatsheet**

&#10230; Masinõppe näpunäidete spikker

<br>

**2. Classification metrics**

&#10230; Klassifikatsiooni mõõdikud

<br>

**3. In a context of a binary classification, here are the main metrics that are important to track in order to assess the performance of the model.**

&#10230; Siin on meil põhi mõõdikud, mille jälgimine on oluline, et mõõta meie mudeli võimekust binaarse klassifikatsiooni kontekstis.

<br>

**4. Confusion matrix ― The confusion matrix is used to have a more complete picture when assessing the performance of a model. It is defined as follows:**

&#10230; Eksimismaatriks - Eksimismaatriksit kasutatakse mudeli võimekust hindamisel täielikuma pildi saamiseks. See on defineeritud järgmiselt:

<br>

**5. [Predicted class, Actual class]**

&#10230; [Ennustatud klass, Tegelik klass]

<br>

**6. Main metrics ― The following metrics are commonly used to assess the performance of classification models:**

&#10230; Põhimõõdikud - järgnevad mõõdikud on tavaliselt kasutatud, et hinnata klassifikatsiooni mudelite võimekust:

<br>

**7. [Metric, Formula, Interpretation]**

&#10230; [Mõõdik, Valem, Tõlgendus]

<br>

**8. Overall performance of model**

&#10230; Mudeli üldine võimekus

<br>

**9. How accurate the positive predictions are**

&#10230; Kui täpsed mudeli positiivsed ennustused on

<br>

**10. Coverage of actual positive sample**

&#10230; Tegeliku positiivse valimi katvus

<br>

**11. Coverage of actual negative sample**

&#10230; Tegeliku negatiivse valimi katvus

<br>

**12. Hybrid metric useful for unbalanced classes**

&#10230; Hübriidmõõdik on kasulik tasakaalustamata klasside jaoks

<br>

**13. ROC ― The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:**

&#10230; ROC - Vastuvõtja töökõver on graafik, millel kuvatakse TPR-i ja FPR-i suhet erineval lävel. Need mõõdikud on kokku võetud allolevas tabelis.

<br>

**14. [Metric, Formula, Equivalent]**

&#10230; [Mõõdik, Valem, Ekvivalent]

<br>

**15. AUC ― The area under the receiving operating curve, also noted AUC or AUROC, is the area below the ROC as shown in the following figure:**

&#10230; AUC - Vastuvõtva töökõvera alune ala, tuntud kui AUC või AUROC, on ROC töökõvera alune ala, mis on näidatud järgmisel joonisel.

<br>

**16. [Actual, Predicted]**

&#10230; [Tegelik, Ennustatud]

<br>

**17. Basic metrics ― Given a regression model f, the following metrics are commonly used to assess the performance of the model:**

&#10230; Põhilised mõõdikud - Arvestades regressioonimudelit f, kasutatakse mudeli võimekuse hindamiseks tavaliselt järgmisi mõõdikuid.

<br>

**18. [Total sum of squares, Explained sum of squares, Residual sum of squares]**

&#10230; [Kogu ruutude summa, selgitatud ruutude summa, jääkliikmete ruutude summa]

<br>

**19. Coefficient of determination ― The coefficient of determination, often noted R2 or r2, provides a measure of how well the observed outcomes are replicated by the model and is defined as follows:**

&#10230; Determinatsioonikordaja - tuntud kui R2 või r2 näitab, kui hästi jälgitavad tulemused mudelis korduvad, ja see on määratletud järgmiselt.

<br>

**20. Main metrics ― The following metrics are commonly used to assess the performance of regression models, by taking into account the number of variables n that they take into consideration:**

&#10230; Põhimõõdikud - Regressioonimudelite võimekuse hindamiseks kasutatakse tavaliselt järgmisi mõõdikuid, võttes arvesse muutujate arvu n, mida nad arvestavad.

<br>

**21. where L is the likelihood and ˆσ2 is an estimate of the variance associated with each response.**

&#10230; kus L on tõenäosus ning ˆσ2 on iga vastusega seotud dispersiooni hinnang. 

<br>

**22. Model selection**

&#10230; Mudeli valik

<br>

**23. Vocabulary ― When selecting a model, we distinguish 3 different parts of the data that we have as follows:**

&#10230; Sõnavara - Mudeli valimisel eristame 3 erinevat osa andmetest järgnevalt:

<br>

**24. [Training set, Validation set, Testing set]**

&#10230; [Treenimiskomplekt, Valideerimiskomplekt, Testimiskomplekt]

<br>

**25. [Model is trained, Model is assessed, Model gives predictions]**

&#10230; [Mudel treenitakse, Mudel valideeritakse, Mudel annab ennustusi]

<br>

**26. [Usually 80% of the dataset, Usually 20% of the dataset]**

&#10230; [Tavaliselt 80% andmekogust, Tavaliselt 20% andmekogust]

<br>

**27. [Also called hold-out or development set, Unseen data]**

&#10230; [Nimetatakse arenduskomplektiks, andmed mida mudel pole näinud]

<br>

**28. Once the model has been chosen, it is trained on the entire dataset and tested on the unseen test set. These are represented in the figure below:**

&#10230; Kui mudel on valitud, treenitakse seda kogu andmehulgaga ja testitakse andmetega mida mudel ei ole näinud. Need on esitatud alloleval joonisel.

<br>

**29. Cross-validation ― Cross-validation, also noted CV, is a method that is used to select a model that does not rely too much on the initial training set. The different types are summed up in the table below:**

&#10230; Ristvalideerimine - Ristvalideerimine, tuntud kui CV, on meetod, mida kasutatakse mudeli valimiseks ning mis ei sõltu liiga palju algsest treeningkomplektist. Erinevad ristkontrolli tüübid on kokku võetud allolevas tabelis:

<br>

**30. [Training on k−1 folds and assessment on the remaining one, Training on n−p observations and assessment on the p remaining ones]**

&#10230; [Treenitakse k-1 ühikuga (treeningkomplektist) ning hindamine ülejäänuga, Treenimine n-p vaatlusega ja hindamine ülejäänud p vaatlustega]

<br>

**31. [Generally k=5 or 10, Case p=1 is called leave-one-out]**

&#10230; [Tavaliselt k=5 või 10, Kui p=1 siis selle nimi on jäta-üks-välja]

<br>

**32. The most commonly used method is called k-fold cross-validation and splits the training data into k folds to validate the model on one fold while training the model on the k−1 other folds, all of this k times. The error is then averaged over the k folds and is named cross-validation error.**

&#10230; Kõige sagedamini kasutatavat meetodit nimetatakse k-kordseks ristvalideerimiseks ja see jagab treeningu andmehulga k kordseks osadeks, et mudelit valideerida ühe osaga, samal ajal treenides mudelit k − 1 teisel osaga, seda kõike k korda. Seejärel arvutatakse viga keskmiselt k korda ja seda nimetatakse ristvalideerimise veaks.

<br>

**33. Regularization ― The regularization procedure aims at avoiding the model to overfit the data and thus deals with high variance issues. The following table sums up the different types of commonly used regularization techniques:**

&#10230; Regulariseerimine - Regulariseerimisprotsessi eesmärk on vältida mudeli ülemäärast ülesobitust treeningandmestikule ning seeläbi tegeleb kõrge dispersiooni probleemidega. Järgmises tabelis on toodud levinumate reguleerimist meetodite tüübid:

<br>

**34. [Shrinks coefficients to 0, Good for variable selection, Makes coefficients smaller, Tradeoff between variable selection and small coefficients]**

&#10230; [Kahandab koefitsiente 0-ni, Hea muutujate valimiseks, Muudab koefitsendid väiksemaks, Muutuja valiku ja väikeste koefitsentide kompromiss]

<br>

**35. Diagnostics**

&#10230; Diagnostika

<br>

**36. Bias ― The bias of a model is the difference between the expected prediction and the correct model that we try to predict for given data points.**

&#10230; Nihe (Vabaliige) - Mudeli vabaliige on erinevus eeldatava ennustuse ja õige mudeli vahel, mida proovime antud andmepunktidega ennustada.

<br>

**37. Variance ― The variance of a model is the variability of the model prediction for given data points.**

&#10230; Variatsioon - Mudeli variatsioon on mudeli ennustuse ja antud andmepunkti varieeruvus.

<br>

**38. Bias/variance tradeoff ― The simpler the model, the higher the bias, and the more complex the model, the higher the variance.**

&#10230; Nihke/Variatsioon kompromiss - Mida lihtsam mudel seda suurem on nihe (vabaliige) ning mida keerulisem on mudel seda suurem on dispersioon.

<br>

**39. [Symptoms, Regression illustration, classification illustration, deep learning illustration, possible remedies]**

&#10230; [Sümptomid, Regressiooni näide, Klassifikatsiooni näide, Süvavõppe näide, võimalikud abinõud]

<br>

**40. [High training error, Training error close to test error, High bias, Training error slightly lower than test error, Very low training error, Training error much lower than test error, High variance]**

&#10230; [Kõrge treenimisviga, Treenimisviga sarnane tesimisvegaga, Kõrge nihe, Treenimisviga madalam kui testimisviga, Väga madal treenimisviga, Treenimisviga palju madalam kui testimisviga, Kõrge variatsioon]

<br>

**41. [Complexify model, Add more features, Train longer, Perform regularization, Get more data]**

&#10230; [Mudeli keerukamaks muutmine, Muutujate lisamine, Pigem treenimine, Regulatsiooni lisamine, Andmehulga suurendamine]

<br>

**42. Error analysis ― Error analysis is analyzing the root cause of the difference in performance between the current and the perfect models.**

&#10230; Veaanalüüs - Veaanalüüs on olemasoleva ja perfektse mudeli võimekuse erinevuse algpõhjuse leidmine.

<br>

**43. Ablative analysis ― Ablative analysis is analyzing the root cause of the difference in performance between the current and the baseline models.**

&#10230; Ablatiivne analüüs - Ablatiivne analüüs on olemasoleva ja baas mudeli võimekuse erinevuse algpõhjuse leidmine.

<br>

**44. Regression metrics**

&#10230; Regressiooni mõõdikud

<br>

**45. [Classification metrics, confusion matrix, accuracy, precision, recall, F1 score, ROC]**

&#10230; [Klassifikatsiooni mõõdikud, Eksimismaatriks, täpsus, saagis, F1 skoor, ROC]

<br>

**46. [Regression metrics, R squared, Mallow's CP, AIC, BIC]**

&#10230; [Regressiooni mõõdikud, R ruudus, Mallow-i CP, AIC, BIC]

<br>

**47. [Model selection, cross-validation, regularization]**

&#10230; [Mudeli valik, ristvalideerimine, reguleerimine]

<br>

**48. [Diagnostics, Bias/variance tradeoff, error/ablative analysis]**

&#10230; [Diagnostika, Nihke/variatsiooni kompromiss, vea/ablatiivne analüüs]
